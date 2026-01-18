from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
def assertGraphContainsExactly(self, graph, kind, num_kind_nodes, consider_subgraphs=False):

    def perform_assert(graph, kind, actual, expected, consider_subgraphs):
        if actual == expected:
            return
        subgraph = 'including' if consider_subgraphs else 'excluding'
        raise AssertionError(f'{graph}\nError: graph contains {actual} {kind} nodes ({subgraph} subgraphs) but expected {expected}')
    if consider_subgraphs:
        strgraph = str(graph)
        count = strgraph.count(kind) - strgraph.count(f'with {kind}')
        perform_assert(graph, kind, count, num_kind_nodes, consider_subgraphs)
        return

    def nodes(block):
        out = []
        for node in block.nodes():
            if node.kind() == kind:
                out.append(node)
            for block in node.blocks():
                out += nodes(block)
        return out
    out_nodes = nodes(graph)
    perform_assert(graph, kind, len(out_nodes), num_kind_nodes, consider_subgraphs)