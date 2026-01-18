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
def handle_placeholder(self, node: torch.fx.Node):
    assert node.op == 'placeholder'
    if isinstance(node.meta['val'], torch.Tensor):
        graph_input = Argument.create(as_tensor=TensorArgument(name=node.name))
        self.graph_state.tensor_values[node.name] = serialize_tensor_meta(node.meta['val'])
    elif isinstance(node.meta['val'], torch.SymInt):
        raise AssertionError('SymInt graph input is not implemented yet.')
    elif isinstance(node.meta['val'], (int, bool, str, float, type(None))):
        graph_input = self.serialize_input(node.meta['val'])
    else:
        raise AssertionError(f'Unimplemented graph input type: {node.meta['val']}')
    self.graph_state.inputs.append(graph_input)