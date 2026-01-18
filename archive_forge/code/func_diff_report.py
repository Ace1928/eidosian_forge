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
def diff_report(self) -> str:
    """Return a string representation of the graph difference.

        The report shows the first pair of nodes that diverges. It also shows the source
        location of the pair of nodes.

        Returns:
            graph_diff_report (str): A string representation of the graph difference.
        """
    graph_a = self.graph_a
    graph_b = self.graph_b
    graph_a_str = str(graph_a)
    graph_b_str = str(graph_b)
    if graph_a_str == graph_b_str:
        return ''
    graph_diff = difflib.ndiff(graph_a_str.splitlines(True), graph_b_str.splitlines(True))
    graph_diff_report = ['Graph diff:', self._indent(''.join(graph_diff))]
    for node_a, node_b in itertools.zip_longest(graph_a.nodes(), graph_b.nodes()):
        if str(node_a) != str(node_b):
            graph_diff_report.append('First diverging operator:')
            node_diff = difflib.ndiff(str(node_a).splitlines(True), str(node_b).splitlines(True))
            source_printout = ['node diff:', self._indent(''.join(node_diff))]
            stack_a = node_a.sourceRange() if node_a else None
            if stack_a:
                source_printout.extend(['Former source location:', self._indent(str(stack_a))])
            stack_b = node_b.sourceRange() if node_b else None
            if stack_b:
                source_printout.extend(['Latter source location:', self._indent(str(stack_b))])
            graph_diff_report.extend(source_printout)
            break
    return '\n'.join(graph_diff_report)