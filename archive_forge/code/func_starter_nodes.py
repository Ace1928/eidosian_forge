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
def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
    """
        Finds nodes that consume module inputs or get_attr nodes.
        """
    starter_cpu_nodes: NodeSet = set()
    starter_acc_nodes: NodeSet = set()
    for node in self.module.graph.nodes:
        if node.op not in {'placeholder', 'get_attr'}:
            continue
        for user in node.users:
            if user in self.acc_nodes:
                starter_acc_nodes.add(user)
            else:
                starter_cpu_nodes.add(user)
    return (starter_cpu_nodes, starter_acc_nodes)