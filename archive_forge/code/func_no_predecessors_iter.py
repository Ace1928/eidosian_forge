import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def no_predecessors_iter(self):
    """Returns an iterator for all nodes with no predecessors."""
    for n in self.nodes:
        if not len(list(self.predecessors(n))):
            yield n