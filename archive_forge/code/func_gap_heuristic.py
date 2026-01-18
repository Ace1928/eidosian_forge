from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import (
def gap_heuristic(height):
    """Apply the gap heuristic."""
    for level in islice(levels, height + 1, max_height + 1):
        for u in level.active:
            R_nodes[u]['height'] = n + 1
        for u in level.inactive:
            R_nodes[u]['height'] = n + 1
        levels[n + 1].active.update(level.active)
        level.active.clear()
        levels[n + 1].inactive.update(level.inactive)
        level.inactive.clear()