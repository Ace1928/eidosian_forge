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
def jit_ignored_properties(module):
    user_annotated_ignored_attributes = getattr(module, '__jit_ignored_attributes__', list())

    def get_properties_names(module):
        return {k for k, v in vars(module).items() if isinstance(v, property)}
    properties = get_properties_names(type(module))
    user_annoted_ignored_properties = set()
    for ignored_attr in user_annotated_ignored_attributes:
        if ignored_attr in properties:
            user_annoted_ignored_properties.add(ignored_attr)
    return user_annoted_ignored_properties