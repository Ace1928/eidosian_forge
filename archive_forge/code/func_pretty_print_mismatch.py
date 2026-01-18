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
def pretty_print_mismatch(self, graph: bool=False):
    """Pretty print details of the mismatch between torch and ONNX.

        Args:
            graph: If True, print the ATen JIT graph and ONNX graph.
        """
    print(f' Mismatch info for graph partition {self.id}: '.center(80, '='))
    if graph:
        print(' ATen JIT graph '.center(80, '='))
        print(self.graph)
        if self._onnx_graph is not None:
            print(' ONNX graph '.center(80, '='))
            print(self._onnx_graph)
    if self.has_mismatch():
        print(' Mismatch error '.center(80, '='))
        print(self.mismatch_error)
    else:
        print(' No mismatch '.center(80, '='))