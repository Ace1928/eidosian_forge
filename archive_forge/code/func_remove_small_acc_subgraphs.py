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
def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
    """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
    result: List[Subgraph] = []
    for subgraph in subgraphs:
        if subgraph.is_acc:
            if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                result.append(subgraph)
            else:
                print(f"Eliminating acc subgraph because it's smaller than the threshold: {len(subgraph.nodes)} < {self.settings.min_acc_module_size}")
                if result:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    subgraph.is_acc = False
                    result.append(subgraph)
        elif result and (not result[-1].is_acc):
            result[-1].nodes.extend(subgraph.nodes)
        else:
            result.append(subgraph)
    return result