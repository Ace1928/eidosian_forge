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
def lazy_bind(concrete_type, unbound_method):
    """
    Return a function that lazily binds `unbound_method` to a provided Module IValue, then invokes the method.

    We do this so that any Python shenanigans that
    will poison type sharing are impossible at compile time.
    """

    def lazy_binding_method(cpp_module, *args):

        def init_fn(script_module):
            orig_class = concrete_type.py_class
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)
            for name, value in concrete_type.get_constants().items():
                setattr(script_module, name, value)
        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        method = types.MethodType(unbound_method, script_module)
        return method(*args)
    lazy_binding_method.original_fn = unbound_method
    lazy_binding_method.__name__ = unbound_method.__name__
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)
    return lazy_binding_method