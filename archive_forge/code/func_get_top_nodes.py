from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_top_nodes(partition: Partition) -> List[Node]:
    """Given a partition, return a list of nodes on the top bfs level"""
    top_nodes: List[Node] = []
    for node in partition.nodes:
        if node.op in {'placeholder', 'get_attr'}:
            continue
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        if not any((n in partition.nodes and n.op not in {'placeholder', 'get_attr'} for n in input_nodes)):
            top_nodes.append(node)
    return top_nodes