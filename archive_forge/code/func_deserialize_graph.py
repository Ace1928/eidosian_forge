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
def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
    for name, tensor_value in serialized_graph.tensor_values.items():
        meta_val = self.deserialize_tensor_meta(tensor_value, self.fake_tensor_mode)
        self.serialized_name_to_meta[name] = meta_val
    for name, sym_int_value in serialized_graph.sym_int_values.items():
        self.serialized_name_to_meta[name] = self.deserialize_sym_int(sym_int_value)
    for name, sym_bool_value in serialized_graph.sym_bool_values.items():
        self.serialized_name_to_meta[name] = self.deserialize_sym_bool(sym_bool_value)
    for input in serialized_graph.inputs:
        placeholder_node = self.graph.placeholder(input.as_tensor.name)
        self.sync_fx_node(input.as_tensor.name, placeholder_node)
    for serialized_node in serialized_graph.nodes:
        try:
            target = self.deserialize_operator(serialized_node.target)
            self.deserialize_node(serialized_node, target)
        except Exception as e:
            raise SerializeError(f'Failed deserializing node {serialized_node}') from e
    outputs = []
    for output in serialized_graph.outputs:
        outputs.append(self.deserialize_graph_output(output))
    if serialized_graph.is_single_tensor_return:
        assert len(outputs) == 1
        outputs = outputs[0]
    else:
        outputs = tuple(outputs)
    output_node = self.graph.output(outputs)
    if serialized_graph.is_single_tensor_return:
        output_node.meta['val'] = output_node.args[0].meta['val']
    else:
        output_node.meta['val'] = tuple((arg.meta['val'] for arg in output_node.args[0]))
    return self.graph