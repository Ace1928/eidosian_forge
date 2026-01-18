import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def reduce_acc_nodes_non_tensor_input(self):
    """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
    non_tensor_cpu_nodes: NodeList = []
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        if node in self.acc_nodes:
            continue
        if is_node_output_tensor(node):
            continue
        non_tensor_cpu_nodes.append(node)
    self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)