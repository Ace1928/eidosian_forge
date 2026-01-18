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
def codegen_kwargs(self):
    if V.graph.cpp_wrapper:
        if self.kwargs and (not self.ordered_kwargs_for_cpp_kernel):
            raise AssertionError('ordered_kwargs_for_cpp_kernel is missing')
        kwargs = []
        for arg_name in self.ordered_kwargs_for_cpp_kernel:
            v = self.get_kwargs_value(arg_name)
            if isinstance(v, sympy.Expr):
                kwargs.append(v)
            else:
                if hasattr(self, 'kwargs_default_value'):
                    type_ = self.kwargs_default_value.get(arg_name).get('type')
                else:
                    type_ = None
                kwargs.append(V.graph.wrapper_code.val_to_cpp_arg_str(type_, v, self.is_legacy_abi_kernel()))
    else:
        kwargs = [f'{k}={V.graph.wrapper_code.val_to_arg_str(v)}' for k, v in self.kwargs.items()]
    return kwargs