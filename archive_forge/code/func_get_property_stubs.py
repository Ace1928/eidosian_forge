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
def get_property_stubs(nn_module):
    """Create property stubs for the properties of the module by creating method stubs for the getter and setter."""
    module_ty = type(nn_module)
    properties_asts = get_class_properties(module_ty, self_name='RecursiveScriptModule')
    rcbs = {}
    for name in dir(module_ty):
        item = getattr(module_ty, name, None)
        if isinstance(item, property):
            if not item.fget:
                raise RuntimeError(f'Property {name} of {nn_module.__name__} must have a getter')
            rcbs[name] = _jit_internal.createResolutionCallbackFromClosure(item.fget)
    stubs = [PropertyStub(rcbs[ast.name().name], ast) for ast in properties_asts]
    return stubs