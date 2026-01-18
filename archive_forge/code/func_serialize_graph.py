import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def serialize_graph(self, graph_module: torch.fx.GraphModule) -> Graph:
    assert isinstance(graph_module, torch.fx.GraphModule)
    for node in graph_module.graph.nodes:
        try:
            getattr(self, f'handle_{node.op}')(node)
        except Exception as e:
            raise SerializeError(f'Failed serializing node {node} in graph: {node.format_node()}') from e
    return Graph(inputs=self.graph_state.inputs, nodes=self.graph_state.nodes, tensor_values=self.graph_state.tensor_values, sym_int_values=self.graph_state.sym_int_values, sym_bool_values=self.graph_state.sym_bool_values, outputs=self.graph_state.outputs, is_single_tensor_return=self.graph_state.is_single_tensor_return)