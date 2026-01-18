import copy
import math
import operator
import traceback
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple
import sympy
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import _ExportPassBase, ProxyValue, PassResult
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._sympy.value_ranges import ValueRanges
def sym_size_cb(proxy, assert_msg, dim):
    dim_proxy = super(_AddRuntimeAssertionsForInlineConstraintsPass, self).call_operator(torch.ops.aten.sym_size.int, (proxy, dim), {}, self._create_dummy_node_metadata())
    cb(proxy=dim_proxy, assert_msg=assert_msg)