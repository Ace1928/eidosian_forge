import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def is_directed_acyclic(self):
    """Returns if this graph is a DAG or not."""
    return nx.is_directed_acyclic_graph(self)