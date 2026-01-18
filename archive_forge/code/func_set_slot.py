import networkx as nx
from collections import deque
def set_slot(self, vertex, n):
    try:
        self.slots[self.index(vertex)] = n
    except ValueError:
        raise ValueError('Vertex is not an end of this edge.')