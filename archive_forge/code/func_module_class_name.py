from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
@property
def module_class_name(self) -> str:
    """Name of the module class.

        E.g. `Embedding`.
        """
    if self._module_class is None:
        return ''
    return self._module_class.__name__