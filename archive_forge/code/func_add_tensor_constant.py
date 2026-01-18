import hashlib
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple
import sympy
import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._logging import LazyString
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import has_free_symbols, ShapeEnv, SymTypes
from torch.utils._mode_utils import no_dispatch
from . import config, ir
from .codegen.common import (
from .codegen.wrapper import CppWrapperCodeGen, CudaWrapperCodeGen, WrapperCodeGen
from .exc import (
from .ir import (
from .lowering import (
from .sizevars import SizeVarAllocator
from .utils import convert_shape_to_inductor, gather_origins, get_sympy_Expr_dtype
from .virtualized import V
def add_tensor_constant(self, data, name=None):

    def allocate(name):
        for constant_name, value in self.constants.items():
            if not data.is_mkldnn and data.size() == value.size() and (data.stride() == value.stride()) and (data.dtype == value.dtype) and (data.device == value.device) and torch.eq(data, value).all():
                return constant_name
        if name is None:
            name = f'constant{len(self.constants)}'
        if name[0].isdigit():
            name = f'constant_{name}'
        prefix = re.sub('[^a-zA-Z0-9_]', '_', name)
        name = prefix
        cnt = 0
        while name in self.constants:
            name = f'{prefix}_{cnt}'
            cnt += 1
        self.constants[name] = data
        self.constant_reprs[name] = hashlib.sha256(repr(data).encode('utf-8')).hexdigest()
        return name
    name = allocate(name)
    return TensorBox.create(ir.ConstantBuffer(name, FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data))))