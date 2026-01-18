import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
def edge_graph(self):
    G = nx.Graph()
    G.add_edges_from([[v.index for v in e.vertices] for e in self.edges])
    return G