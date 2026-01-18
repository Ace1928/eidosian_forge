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
def is_same_module_as(self, node: _IRNode) -> bool:
    """Determines if the provided node pertains to the same module as this node."""
    return self.stack_meta == node.stack_meta