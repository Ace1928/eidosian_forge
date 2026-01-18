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
def essential_node_kinds(self) -> Set[str]:
    """Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
    return {n.kind() for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS}