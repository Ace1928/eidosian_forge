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
def get_read_indices(r):
    cb = ComputedBuffer(name=None, layout=FlexibleLayout(device=r.get_device(), dtype=r.get_dtype(), size=r.get_size()), data=r)
    read_writes = cb.get_read_writes()
    range_vars = [r for r in read_writes.range_vars if isinstance(r, sympy.Expr) and (not isinstance(r, sympy.Number))]
    indices = []
    changed = False
    for md in sorted(read_writes.reads, key=lambda x: x.name):
        if all((r in md.index.free_symbols for r in range_vars)):
            indices.append(md.index)
            if md.name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[md.name]
                original_stride = buf.layout.stride
                buf.decide_layout()
                if buf.layout.stride != original_stride:
                    changed = True
    return (indices, changed)