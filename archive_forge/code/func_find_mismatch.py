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
def find_mismatch(self, options: Optional[VerificationOptions]=None):
    """
        Find all mismatches between the TorchScript IR graph and the exported onnx model.

        Binary searches the model graph to find the minimal subgraph that exhibits the
        mismatch. A `GraphInfo` object is created for each subgraph, recording the test
        inputs and export options, as well as the validation results.

        Args:
            options: The verification options.
        """
    self.clear()
    if options is None:
        options = VerificationOptions()
    if self.export_options.verbose:
        print(self.graph)
    if len(list(self.graph.outputs())) == 0:
        return
    assert len(self.input_args) + len(self.params_dict) == len(list(self.graph.inputs())), f'Number of graph inputs({len(list(self.graph.inputs()))}) does not match the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)}).'
    self.mismatch_error, self._onnx_graph, self.pt_outs, _ = self.verify_export(options)
    if self.mismatch_error is None:
        return
    if self.essential_node_count() <= 1:
        return
    full_kwargs = {k.debugName(): v for k, v in zip(self.graph.inputs(), self.input_args)}
    full_params = self.params_dict
    upper_graph = self._partition_upper_graph()
    upper_args, upper_params = self._args_and_params_for_partition_graph(upper_graph, {}, full_kwargs, full_params)
    self.upper_graph_info = GraphInfo(upper_graph, upper_args, upper_params, self.export_options, id=self.id + '0')
    self.upper_graph_info.find_mismatch(options)
    bridge_kwargs = self.upper_graph_info._bridge_kwargs()
    lower_graph = self._partition_lower_graph()
    lower_args, lower_params = self._args_and_params_for_partition_graph(lower_graph, bridge_kwargs, full_kwargs, full_params)
    self.lower_graph_info = GraphInfo(lower_graph, lower_args, lower_params, self.export_options, id=self.id + '1')
    self.lower_graph_info.find_mismatch(options)