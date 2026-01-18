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
def ban_recomputation(node):
    if 'recompute' in node.meta:
        return node.meta['recompute'] == 0
    elif AGGRESSIVE_RECOMPUTATION:
        return node.op == 'call_function' and get_aten_target(node) in unrecomputable_ops
    else:
        if node.op != 'call_function':
            return False
        if get_aten_target(node) not in recomputable_ops:
            return True
        if node.target == operator.getitem:
            return False
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return False
        if is_materialized_backwards(node):
            return True
        if not graph_has_recomputable_ops:
            if compiler == 'inductor' and node.dist_from_bw > config.max_dist_from_bw:
                return True
        input_tensors_size = sum((_size_of(i) for i in node.args if isinstance(i, fx.Node)))
        output_size = _size_of(node)
        return output_size * 4 < input_tensors_size