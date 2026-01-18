from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@contextlib.contextmanager
@_beartype.beartype
def exporter_context(model, mode: _C_onnx.TrainingMode, verbose: bool):
    with select_model_mode_for_export(model, mode) as mode_ctx, disable_apex_o2_state_dict_hook(model) as apex_ctx, setup_onnx_logging(verbose) as log_ctx, diagnostics.create_export_diagnostic_context() as diagnostic_ctx:
        yield (mode_ctx, apex_ctx, log_ctx, diagnostic_ctx)