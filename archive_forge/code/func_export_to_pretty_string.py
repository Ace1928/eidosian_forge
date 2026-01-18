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
@_beartype.beartype
@torch._disable_dynamo
def export_to_pretty_string(model, args, export_params=True, verbose=False, training=_C_onnx.TrainingMode.EVAL, input_names=None, output_names=None, operator_export_type=_C_onnx.OperatorExportTypes.ONNX, export_type=None, google_printer=False, opset_version=None, keep_initializers_as_inputs=None, custom_opsets=None, add_node_names=True, do_constant_folding=True, dynamic_axes=None):
    """
    Similar to :func:`export`, but returns a text representation of the ONNX
    model. Only differences in args listed below. All other args are the same
    as :func:`export`.

    Args:
        add_node_names (bool, default True): Whether or not to set
            NodeProto.name. This makes no difference unless
            ``google_printer=True``.
        google_printer (bool, default False): If False, will return a custom,
            compact representation of the model. If True will return the
            protobuf's `Message::DebugString()`, which is more verbose.

    Returns:
        A UTF-8 str containing a human-readable representation of the ONNX model.
    """
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    if custom_opsets is None:
        custom_opsets = {}
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type
    with exporter_context(model, training, verbose):
        val_keep_init_as_ip = _decide_keep_init_as_input(keep_initializers_as_inputs, operator_export_type, opset_version)
        val_add_node_names = _decide_add_node_names(add_node_names, operator_export_type)
        val_do_constant_folding = _decide_constant_folding(do_constant_folding, operator_export_type, training)
        args = _decide_input_format(model, args)
        graph, params_dict, torch_out = _model_to_graph(model, args, verbose, input_names, output_names, operator_export_type, val_do_constant_folding, training=training, dynamic_axes=dynamic_axes)
        return graph._pretty_print_onnx(params_dict, opset_version, False, operator_export_type, google_printer, val_keep_init_as_ip, custom_opsets, val_add_node_names)