from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def residual_capacity(self, i, p):
    """Returns the residual capacity of an edge i in the direction away
        from its endpoint p.
        """
    return self.edge_capacities[i] - self.edge_flow[i] if self.edge_sources[i] == p else self.edge_flow[i]