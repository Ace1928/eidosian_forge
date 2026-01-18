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
def draw_joint_graph(graph: torch.fx.GraphModule, joint_inputs, file_name: str='full_graph.png', dot_graph_shape: Optional[str]=None):
    draw_graph(graph, file_name, dot_graph_shape=dot_graph_shape)
    return default_partition(graph, joint_inputs)