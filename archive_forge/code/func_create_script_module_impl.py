import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    method_stubs = stubs_fn(nn_module)
    property_stubs = get_property_stubs(nn_module)
    hook_stubs, pre_hook_stubs = get_hook_stubs(nn_module)
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    ignored_properties = jit_ignored_properties(nn_module)

    def init_fn(script_module):
        for name in concrete_type.get_attributes().keys():
            orig_value = getattr(nn_module, name)
            orig_value = orig_value.value if isinstance(orig_value, torch.jit.Attribute) else orig_value
            cpp_module.setattr(name, orig_value)
        for name, sub_concrete_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            assert isinstance(orig_value, Module), f'Expected Module but got {type(orig_value)}'
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                scripted = orig_value
            else:
                scripted = create_script_module_impl(orig_value, sub_concrete_type, stubs_fn)
            cpp_module.setattr(name, scripted)
            script_module._modules[name] = scripted
        for name in dir(nn_module):
            if name in ignored_properties:
                continue
            item = getattr(nn_module, name, None)
            if inspect.ismethod(item) and _jit_internal.is_ignored_fn(item):
                unbound_function = getattr(nn_module, name).__func__
                bound_method = unbound_function.__get__(script_module)
                setattr(script_module, name, bound_method)
            elif concrete_type.is_ignored_attribute(name):
                setattr(script_module, name, item)
        script_module._concrete_type = concrete_type
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
    if concrete_type not in concrete_type_store.methods_compiled:
        create_methods_and_properties_from_stubs(concrete_type, method_stubs, property_stubs)
        create_hooks_from_stubs(concrete_type, hook_stubs, pre_hook_stubs)
        torch._C._run_emit_module_hook(cpp_module)
        concrete_type_store.methods_compiled.add(concrete_type)
    for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
        script_module._forward_pre_hooks[idx] = fn
    for idx, fn in enumerate(script_module._c._get_forward_hooks()):
        script_module._forward_hooks[idx] = fn
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)) and '__len__' not in cpp_module._method_names():
        script_module.define(f'def __len__(self):\n   return {len(nn_module)}\n')
    if isinstance(nn_module, torch.nn.ModuleDict) and '__contains__' not in cpp_module._method_names():
        if len(nn_module.keys()):
            keys = repr(list(nn_module.keys()))
            script_module.define(f'def __contains__(self, key: str):\n   return key in {keys}\n')
        else:
            script_module.define('def __contains__(self, key: str):\n   return False\n')
    for method_stub in method_stubs:
        if method_stub.original_method is None:
            continue
        name = method_stub.original_method.__name__
        if name != method_stub.def_.name().name:
            continue
        script_method = cpp_module._get_method(name)
        wrapped_script_method = functools.wraps(method_stub.original_method)(script_method)
        script_module.__dict__[name] = wrapped_script_method
    for property_stub in property_stubs:
        property_name = property_stub.def_.name().name
        fget = cpp_module._get_method(property_stub.def_.getter_name().name)
        setter_name = property_stub.def_.setter_name()
        fset = cpp_module._get_method(setter_name.name) if setter_name else None
        script_module.__dict__[property_name] = property(property_name, fget, fset)
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER:
            add_python_attr_to_scripted_model(script_module, nn_module, name)
    return script_module