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
def get_module_concrete_type(nn_module, share_types=True):
    """
    Get a concrete type for nn_modules.

    If share_types is True, the concrete type is fetched from concrete_type_store.
    If it is False, a new concrete type is created without first searching concrete_type_store.

    Args:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        share_types = Whether to share underlying JIT types between modules (if possible).

    Returns:
        A concrete type for nn_module.
    """
    assert isinstance(nn_module, Module)
    if isinstance(nn_module, torch.jit.ScriptModule) and hasattr(nn_module, '_concrete_type'):
        return nn_module._concrete_type
    if share_types:
        concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    else:
        concrete_type_builder = infer_concrete_type_builder(nn_module, share_types)
        concrete_type_builder.set_poisoned()
        concrete_type = concrete_type_builder.build()
    return concrete_type