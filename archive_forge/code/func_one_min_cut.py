import networkx as nx
from collections import deque
def one_min_cut(self, source, sink):
    capacity = {e: e.multiplicity for e in self.edges}
    cut = Graph.one_min_cut(self, source, sink, capacity)
    cut['size'] = sum((e.multiplicity for e in cut['edges']))
    return cut