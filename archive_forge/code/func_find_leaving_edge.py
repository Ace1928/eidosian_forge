from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def find_leaving_edge(self, Wn, We):
    """Returns the leaving edge in a cycle represented by Wn and We."""
    j, s = min(zip(reversed(We), reversed(Wn)), key=lambda i_p: self.residual_capacity(*i_p))
    t = self.edge_targets[j] if self.edge_sources[j] == s else self.edge_sources[j]
    return (j, s, t)