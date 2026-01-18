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
@_beartype.beartype
def save_diagnostics(self, destination: str) -> None:
    """Saves the export diagnostics as a SARIF log to the specified destination path.

        Args:
            destination: The destination to save the diagnostics SARIF log.
                It must have a `.sarif` extension.

        Raises:
            ValueError: If the destination path does not end with `.sarif` extension.
        """
    if not destination.endswith('.sarif'):
        message = f"'destination' must have a .sarif extension, got {destination}"
        log.fatal(message)
        raise ValueError(message)
    self.diagnostic_context.dump(destination)