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
def compile_unbound_method(concrete_type, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn, fn.__name__)
    with torch._jit_internal._disable_emit_hooks():
        create_methods_and_properties_from_stubs(concrete_type, (stub,), ())
    return stub