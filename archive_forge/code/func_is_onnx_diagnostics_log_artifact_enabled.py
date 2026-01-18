from __future__ import annotations
import dataclasses
import functools
import logging
from typing import Any, Optional
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]
import torch
import torch.fx
from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import decorator, formatter
from torch.onnx._internal.fx import registration, type_utils as fx_type_utils
def is_onnx_diagnostics_log_artifact_enabled() -> bool:
    return torch._logging._internal.log_state.is_artifact_enabled(_ONNX_DIAGNOSTICS_ARTIFACT_LOGGER_NAME)