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
def find_deps(self) -> Dict[torch.fx.Node, NodeSet]:
    """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
    deps: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        for user in node.users:
            if user.op != 'output':
                deps[user].add(node)
    return deps