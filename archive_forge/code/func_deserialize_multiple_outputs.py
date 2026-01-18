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
def deserialize_multiple_outputs(self, serialized_node: Node, fx_node: torch.fx.Node) -> None:
    deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

    def generate_getitem(meta_val, fx_node: torch.fx.Node, arg: TensorArgument, idx: int):
        name = arg.name
        individual_output = self.graph.create_node('call_function', operator.getitem, (fx_node, idx), name=name)
        self.sync_fx_node(name, individual_output)
        meta_val.append(self.serialized_name_to_meta[name])
        individual_output.meta.update(deserialized_metadata)

    def generate_getitems(meta_val, fx_node: torch.fx.Node, args):
        for idx, arg in enumerate(args):
            if isinstance(arg, Argument):
                arg = arg.value
            if isinstance(arg, TensorArgument):
                generate_getitem(meta_val, fx_node, arg, idx)
            elif isinstance(arg, (list, tuple)):
                list_output = self.graph.create_node('call_function', operator.getitem, (fx_node, idx))
                meta_val.append([])
                generate_getitems(meta_val[-1], list_output, arg)
                list_output.meta.update(deserialized_metadata)
                list_output.meta['val'] = meta_val[-1]
            else:
                raise NotImplementedError(f'Unimplemented node output type: {arg}')
    meta_val: List[Any] = []
    if len(serialized_node.outputs) == 1:
        assert isinstance(serialized_node.outputs[0].value, list)
        assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
        generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors)
    else:
        generate_getitems(meta_val, fx_node, serialized_node.outputs)
    fx_node.meta['val'] = tuple(meta_val)
    self.serialized_name_to_node[fx_node.name] = fx_node