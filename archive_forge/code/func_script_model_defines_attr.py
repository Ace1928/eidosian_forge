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
def script_model_defines_attr(script_model, attr):
    script_attr = getattr(script_model, attr, None)
    if script_attr is None:
        return False
    default_attr = getattr(torch.jit.RecursiveScriptModule, attr, None)
    if default_attr is None:
        return False
    return script_attr != default_attr