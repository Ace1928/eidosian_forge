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
def reduce_acc_nodes_non_tensor_output(self):
    """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
    while True:
        new_cpu_nodes: NodeList = []
        for acc_node in self.acc_nodes:
            if is_node_output_tensor(acc_node):
                continue
            for user in acc_node.users:
                if user not in self.acc_nodes:
                    new_cpu_nodes.append(acc_node)
                    break
        if not new_cpu_nodes:
            break
        for new_cpu_node in new_cpu_nodes:
            self.acc_nodes.remove(new_cpu_node)
        self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)