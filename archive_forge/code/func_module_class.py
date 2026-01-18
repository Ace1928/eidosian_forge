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
def module_class(self) -> Optional[type]:
    """Returns the module class of the top module."""
    return self.top()._module_class