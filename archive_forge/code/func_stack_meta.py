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
def stack_meta(self) -> _ModuleStackMeta:
    """Returns the module stack meta data associated with this node."""
    return self._stack_meta