from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import (
def relabel(u):
    """Relabel a node to create an admissible edge."""
    grt.add_work(len(R_succ[u]))
    return min((R_nodes[v]['height'] for v, attr in R_succ[u].items() if attr['flow'] < attr['capacity'])) + 1