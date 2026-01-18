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
def codegen_list_tuple_access(self, basename, indices):
    if len(indices) > 0:
        itype, i = indices[0]
        if itype == list:
            return self.codegen_list_tuple_access(f'{basename}[{i}]', indices[1:])
        elif itype == tuple:
            tuple_access = V.graph.wrapper_code.codegen_tuple_access(basename, self.get_name(), str(i))
            return self.codegen_list_tuple_access(tuple_access, indices[1:])
        elif itype == dict:
            return self.codegen_list_tuple_access(f"{basename}['{i}']", indices[1:])
        else:
            raise AssertionError('non supported index type')
    else:
        return basename