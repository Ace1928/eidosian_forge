import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
def export_extern_kernel_node(self):
    assert isinstance(self, FallbackKernel)
    args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
    ordered_kwargs = [kwargs.get(key, None) for key in self.ordered_kwargs_for_cpp_kernel]
    serializer = GraphModuleSerializer(None, None)
    named_arguments = serializer.serialize_inputs(self.op_overload, args, kwargs)

    def handle_single_output(return_type, output):
        if isinstance(return_type, torch.TensorType):
            out = output
            if isinstance(output, (list, tuple)):
                assert len(output) == 1
                out = output[0]
            return export_schema.Argument.create(as_tensor=export_schema.TensorArgument(name=out.get_name()))
        elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.TensorType):
            return export_schema.Argument.create(as_tensors=[export_schema.TensorArgument(name=out.get_name()) for out in output])
        else:
            raise RuntimeError(f'Unsupported return type {type(return_type)}')
    target = self.op_overload
    returns = target._schema.returns
    if len(returns) == 1:
        return_type = returns[0].real_type
        output_arguments = [handle_single_output(return_type, self.outputs)]
    else:
        assert isinstance(self.outputs, tuple)
        assert len(returns) == len(self.outputs)
        output_arguments = [handle_single_output(return_schema.real_type, output) for return_schema, output in zip(returns, self.outputs)]
    node = ExternKernelNode(name=self.get_name(), node=export_schema.Node(target=self.op_overload.name(), inputs=named_arguments, outputs=output_arguments, metadata={}))
    V.graph.extern_kernel_nodes.append(node)
    return [*args, *ordered_kwargs]