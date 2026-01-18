from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def check_training_mode(op_train_mode: int, op_name: str) -> None:
    """Warns the user if the model's training mode and the export mode do not agree."""
    if GLOBALS.training_mode == _C_onnx.TrainingMode.PRESERVE:
        return
    if op_train_mode:
        op_mode_enum = _C_onnx.TrainingMode.TRAINING
    else:
        op_mode_enum = _C_onnx.TrainingMode.EVAL
    if op_mode_enum == GLOBALS.training_mode:
        return
    op_mode_text = f'train={bool(op_train_mode)}'
    warnings.warn(f"ONNX export mode is set to {GLOBALS.training_mode}, but operator '{op_name}' is set to {op_mode_text}. Exporting with {op_mode_text}.")