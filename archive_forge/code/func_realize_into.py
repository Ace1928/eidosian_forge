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
@classmethod
def realize_into(cls, src, dst):
    if not isinstance(dst, ReinterpretView):
        if is_storage_and_layout(dst):
            storage, layout = as_storage_and_layout(dst)
            dst = ReinterpretView(storage, layout)
    assert isinstance(dst, ReinterpretView), dst
    if isinstance(src, TensorBox):
        return cls.realize_into(src.data, dst)
    if isinstance(src, StorageBox):
        src.realize()
        assert hasattr(src.data, 'layout')
        if cls.can_realize_into_without_copy(src):
            src.data.layout = AliasedLayout(dst)
            return src.data
    pw = Pointwise.create(device=src.get_device(), dtype=src.get_dtype(), inner_fn=src.make_loader(), ranges=[V.graph.sizevars.guard_equals(a, b) for a, b in zip(src.get_size(), dst.get_size())])
    return cls.realize_into(pw, dst)