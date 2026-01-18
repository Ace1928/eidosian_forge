from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_extra_size_of(node: Node, nodes: Set[Node]) -> int:
    """Given a node and a set of nodes,
    this function return the extra size that needed
    if this node is included in this set.
    """
    input_nodes: Dict[Node, None] = {}
    map_arg(node.args, input_nodes.setdefault)
    map_arg(node.kwargs, input_nodes.setdefault)
    total_size_of_input_nodes = 0
    for n in input_nodes:
        if n not in nodes:
            size_bytes = getattr(n, 'size_bytes', None)
            if size_bytes:
                total_size_of_input_nodes += size_bytes.output_size
            else:
                raise RuntimeError('node has no size_bytes attr')
    size_bytes = getattr(node, 'size_bytes', None)
    if size_bytes:
        total_size_of_input_nodes += size_bytes.total_size
    else:
        raise RuntimeError('node has no size_bytes attr')
    return total_size_of_input_nodes