from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def maybe_merge_partition(self_id: int, other_id: int):
    merged_nodes = copy(partitions_by_id[self_id].nodes)
    merged_nodes.update(partitions_by_id[other_id].nodes)
    visited: Set[Node] = set()

    def dfs_iter_find_cycle(root_node):
        stack: Deque[Node] = deque()
        stack.append(root_node)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            if node in merged_nodes:
                return True
            if node in assignment:
                for p_node in partitions_by_id[assignment[node]].nodes:
                    for user_node in p_node.users:
                        if user_node not in partitions_by_id[assignment[node]].nodes:
                            stack.append(user_node)
            else:
                for user_node in node.users:
                    stack.append(user_node)
            visited.add(node)
        return False
    for node in merged_nodes:
        for user_node in node.users:
            if user_node not in merged_nodes and dfs_iter_find_cycle(user_node):
                return False
    partitions_by_id[self_id].nodes = merged_nodes
    for node in partitions_by_id[other_id].nodes:
        assignment[node] = self_id
    del partitions_by_id[other_id]
    return True