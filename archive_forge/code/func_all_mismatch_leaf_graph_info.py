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
def all_mismatch_leaf_graph_info(self) -> List['GraphInfo']:
    """Return a list of all leaf `GraphInfo` objects that have mismatch."""
    if not self.has_mismatch():
        return []
    no_mismatch_children = (self.upper_graph_info is None or not self.upper_graph_info.has_mismatch()) and (self.lower_graph_info is None or not self.lower_graph_info.has_mismatch())
    if no_mismatch_children:
        return [self]
    results = []
    if self.upper_graph_info is not None:
        results += self.upper_graph_info.all_mismatch_leaf_graph_info()
    if self.lower_graph_info is not None:
        results += self.lower_graph_info.all_mismatch_leaf_graph_info()
    return results