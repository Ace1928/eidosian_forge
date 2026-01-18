from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def reduced_cost(self, i):
    """Returns the reduced cost of an edge i."""
    c = self.edge_weights[i] - self.node_potentials[self.edge_sources[i]] + self.node_potentials[self.edge_targets[i]]
    return c if self.edge_flow[i] == 0 else -c