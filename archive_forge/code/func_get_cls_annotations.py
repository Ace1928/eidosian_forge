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
def get_cls_annotations(cls):
    cls_annotations = inspect.get_annotations(cls)
    if cls_annotations:
        return cls_annotations
    for base in cls.__bases__:
        cls_annotations = get_cls_annotations(base)
        if cls_annotations:
            return cls_annotations
    return {}