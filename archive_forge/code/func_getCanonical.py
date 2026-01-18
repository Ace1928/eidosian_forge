import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def getCanonical():
    G = nx.Graph()
    G.add_node('A', label='A')
    G.add_node('B', label='B')
    G.add_node('C', label='C')
    G.add_node('D', label='D')
    G.add_edge('A', 'B', label='a-b')
    G.add_edge('B', 'C', label='b-c')
    G.add_edge('B', 'D', label='b-d')
    return G