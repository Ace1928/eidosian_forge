from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@_beartype.beartype
def compare_onnx_pytorch_model_with_input(input_args, input_kwargs):
    pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
    pt_model_copy = _try_clone_model(pt_model)
    pt_outs = pt_model_copy(*pt_args, **pt_kwargs)
    onnx_inputs = _prepare_input_for_onnx(input_args, input_kwargs, options.remained_onnx_input_idx, options.flatten)
    onnx_outs = _run_onnx(onnx_session, onnx_inputs)
    _compare_onnx_pytorch_outputs(onnx_outs=onnx_outs, pt_outs=pt_outs, options=options)