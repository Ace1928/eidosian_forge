from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
def find_max_weight_clique(self):
    """Find a maximum weight clique."""
    nodes = sorted(self.G.nodes(), key=lambda v: self.G.degree(v), reverse=True)
    nodes = [v for v in nodes if self.node_weights[v] > 0]
    self.expand([], 0, nodes)