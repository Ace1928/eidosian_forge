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
def make_stubs_from_exported_methods(mod):
    stubs = []
    for name in dir(mod):
        item = getattr(mod, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            stubs.append(make_stub_from_method(mod, name))
    return stubs