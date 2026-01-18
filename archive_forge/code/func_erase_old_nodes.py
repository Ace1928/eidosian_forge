import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def erase_old_nodes(self, graph: torch.fx.GraphModule, split_node: torch.fx.Node, next_users: List[torch.fx.Node]):
    to_remove = [split_node]
    counters['inductor']['scmerge_split_removed'] += 1
    for getitem_node in split_node.users.keys():
        to_remove.append(getitem_node)
    for next_user in next_users:
        if next_user.target not in {torch.cat, torch.stack}:
            continue
        counters['inductor']['scmerge_cat_removed'] += 1
        to_remove.append(next_user)
    for node in reversed(to_remove):
        graph.erase_node(node)