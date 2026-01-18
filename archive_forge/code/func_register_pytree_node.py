from __future__ import annotations
import contextlib
import functools
import inspect
from typing import (
import torch._dynamo
import torch.export as torch_export
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree
@_beartype.beartype
def register_pytree_node(self, class_type: Type, flatten_func: pytree.FlattenFunc, unflatten_func: pytree.UnflattenFunc):
    """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
    if class_type in pytree.SUPPORTED_NODES or class_type in self._extensions:
        return
    self._extensions[class_type] = (flatten_func, unflatten_func)