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
def infer_methods_to_compile(nn_module):
    """Implement the default rules for which methods should act as starting points for compilation.

    (TODO add a link when the rules are published).
    """
    check_module_initialized(nn_module)
    user_annotated_ignored_attributes = getattr(nn_module, '__jit_ignored_attributes__', list())
    ignored_properties = jit_ignored_properties(nn_module)
    methods: List[str] = []
    if hasattr(nn_module, 'forward') and (not _jit_internal.is_ignored_fn(nn_module.forward)):
        forward_func = getattr(nn_module.forward, '__func__', None)
        module_forward = getattr(torch.nn.Module, 'forward', None)
        if forward_func != module_forward:
            methods = ['forward']
    exported = []
    for name in dir(nn_module):
        if name in ignored_properties:
            continue
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            exported.append(name)
    methods = methods + exported
    overload_name_mappings = dict(getattr(nn_module, '__overloads__', {}))
    overload_info = get_overload_annotations(nn_module, ignored_properties)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)
    nn_module.__overloads__ = overload_name_mappings

    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings
    filtered_methods = filter(ignore_overloaded, methods)
    uniquer: Set[str] = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)
    stubs = []
    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs