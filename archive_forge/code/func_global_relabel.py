from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import (
def global_relabel(from_sink):
    """Apply the global relabeling heuristic."""
    src = t if from_sink else s
    heights = reverse_bfs(src)
    if not from_sink:
        del heights[t]
    max_height = max(heights.values())
    if from_sink:
        for u in R:
            if u not in heights and R_nodes[u]['height'] < n:
                heights[u] = n + 1
    else:
        for u in heights:
            heights[u] += n
        max_height += n
    del heights[src]
    for u, new_height in heights.items():
        old_height = R_nodes[u]['height']
        if new_height != old_height:
            if u in levels[old_height].active:
                levels[old_height].active.remove(u)
                levels[new_height].active.add(u)
            else:
                levels[old_height].inactive.remove(u)
                levels[new_height].inactive.add(u)
            R_nodes[u]['height'] = new_height
    return max_height