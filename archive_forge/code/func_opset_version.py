from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
@property
def opset_version(self) -> int:
    """The ONNX opset version the exporter should target. Defaults to the latest
        supported ONNX opset version: 18. The default version will increment over time as
        ONNX continues to evolve."""
    return self._opset_version