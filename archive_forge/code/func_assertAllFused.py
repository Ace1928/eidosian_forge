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
def assertAllFused(self, graph, except_for=()):

    def get_nodes_and_parents_recursively(block, kind, acc):
        for node in block.nodes():
            if node.kind() == kind:
                acc[block].append(node)
            elif node.kind() == 'prim::DifferentiableGraph':
                get_nodes_and_parents_recursively(node.g('Subgraph'), kind, acc)
            elif node.kind() == 'prim::If' and (node.inputs().__next__().node().kind() == 'aten::all' or node.inputs().__next__().node().kind() == 'prim::TypeCheck' or node.inputs().__next__().node().kind() == 'prim::RequiresGradCheck'):
                get_nodes_and_parents_recursively(node.blocks().__next__(), kind, acc)
            else:
                for inner_block in node.blocks():
                    get_nodes_and_parents_recursively(inner_block, kind, acc)
    allowed_nodes = {'prim::Constant', FUSION_GROUP, 'prim::BailoutTemplate', 'prim::TupleConstruct', 'prim::If', 'prim::TypeCheck', 'prim::RequiresGradCheck'} | set(except_for)
    fusion_groups: Dict[torch._C.Block, List[torch._C.Node]] = defaultdict(list)
    get_nodes_and_parents_recursively(graph, FUSION_GROUP, fusion_groups)
    self.assertTrue(len(fusion_groups) == 1, f'got {graph}')
    graph, fusion_nodes = next(iter(fusion_groups.items()))
    self.assertTrue(len(fusion_nodes) == 1, f'got {graph}')
    self.assertTrue(all((node.kind() in allowed_nodes for node in graph.nodes())), f'got {graph}')