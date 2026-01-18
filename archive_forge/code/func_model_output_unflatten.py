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
def model_output_unflatten(values: List[Any], context: pytree.Context) -> modeling_outputs.ModelOutput:
    output_type, keys = context
    return output_type(**dict(zip(keys, values)))