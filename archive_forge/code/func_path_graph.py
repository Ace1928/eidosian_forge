from itertools import combinations
import pytest
import networkx as nx
def path_graph():
    """Return a path graph of length three."""
    G = nx.path_graph(3, create_using=nx.DiGraph)
    G.graph['name'] = 'path'
    nx.freeze(G)
    return G