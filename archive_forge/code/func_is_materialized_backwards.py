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
def is_materialized_backwards(node):
    cur_nodes = {node}
    while len(cur_nodes) > 0:
        cur = cur_nodes.pop()
        for user in cur.users:
            if user not in required_fw_nodes and (not is_fusible(cur, user)):
                return True
            if user not in required_fw_nodes and get_aten_target(user) in view_ops:
                cur_nodes.add(user)
    return False