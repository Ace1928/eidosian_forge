import itertools
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
def neighbors(v):
    return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])