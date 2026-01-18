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
def pretty_print_tree(self):
    """Pretty print `GraphInfo` tree.

        Each node represents a subgraph, showing the number of nodes in the subgraph and
        a check mark if the subgraph has output mismatch between torch and ONNX.

        The id of the subgraph is shown under the node. The `GraphInfo` object for any
        subgraph can be retrieved by calling `graph_info.find_partition(id)`.

        Example::

            ==================================== Tree: =====================================
            5 X   __2 X    __1 ✓
            id:  |  id: 0 |  id: 00
                 |        |
                 |        |__1 X (aten::relu)
                 |           id: 01
                 |
                 |__3 X    __1 ✓
                    id: 1 |  id: 10
                          |
                          |__2 X     __1 X (aten::relu)
                             id: 11 |  id: 110
                                    |
                                    |__1 ✓
                                       id: 111
            =========================== Mismatch leaf subgraphs: ===========================
            ['01', '110']
            ============================= Mismatch node kinds: =============================
            {'aten::relu': 2}

        """
    GraphInfoPrettyPrinter(self).pretty_print()