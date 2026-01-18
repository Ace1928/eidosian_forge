from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def initialize_spanning_tree(self, n, faux_inf):
    self.edge_count = len(self.edge_indices)
    self.edge_flow = list(chain(repeat(0, self.edge_count), (abs(d) for d in self.node_demands)))
    self.node_potentials = [faux_inf if d <= 0 else -faux_inf for d in self.node_demands]
    self.parent = list(chain(repeat(-1, n), [None]))
    self.parent_edge = list(range(self.edge_count, self.edge_count + n))
    self.subtree_size = list(chain(repeat(1, n), [n + 1]))
    self.next_node_dft = list(chain(range(1, n), [-1, 0]))
    self.prev_node_dft = list(range(-1, n))
    self.last_descendent_dft = list(chain(range(n), [n - 1]))
    self._spanning_tree_initialized = True