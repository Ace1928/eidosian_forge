from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def get_node_weight(node) -> int:
    mem_sz = _size_of(node)
    mem_sz = int(mem_sz * 1.1 ** max(min(node.dist_from_bw, 100), 1))
    if is_materialized(node):
        return mem_sz
    else:
        return mem_sz * 2