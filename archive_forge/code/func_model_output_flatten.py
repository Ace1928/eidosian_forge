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
def model_output_flatten(output: modeling_outputs.ModelOutput) -> Tuple[List[Any], pytree.Context]:
    return (list(output.values()), (type(output), list(output.keys())))