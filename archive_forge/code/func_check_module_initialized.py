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
def check_module_initialized(mod):
    assert isinstance(mod, torch.nn.Module)
    if not hasattr(mod, '_parameters'):
        raise RuntimeError(f"'{torch.typename(type(mod))}' has not been initialized, did you forget to call 'super()'?")
    if not hasattr(mod, 'remote_parameters'):
        for name, param in mod._parameters.items():
            if param is not None and torch.nn.parameter.is_lazy(param):
                raise RuntimeError("'{}' has uninitialized parameters {}. Did you forget to run a forward pass?".format(torch.typename(type(mod)), name))
        for name, buf in mod._buffers.items():
            if buf is not None and torch.nn.parameter.is_lazy(buf):
                raise RuntimeError("'{}' has uninitialized buffers {}. Did you forget to run a forward pass?".format(torch.typename(type(mod)), name))