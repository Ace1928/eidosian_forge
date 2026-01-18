from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace the graph of the given GraphModule with one that contains the same nodes as the
    original, but in topologically sorted order.

    This is used by the merge_matmul transformation below, which disturbs the topologically sorted
    order of its input GraphModule, so that this order is restored before further transformation.

    Arguments:
        gm: The graph module to topologically sort. It is modified in-place.

    Returns:
        The graph module in-place sorted
    """
    indeg = {node: 0 for node in gm.graph.nodes}
    new_graph = torch.fx.Graph()
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1
    queue: collections.deque = collections.deque()
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            queue.append(node)
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    while len(queue) > 0:
        cur = queue.popleft()
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                queue.append(user)
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(f'Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}')
    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm