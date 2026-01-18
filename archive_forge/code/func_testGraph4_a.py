import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def testGraph4_a(self):
    G1 = getCanonical()
    G2 = nx.Graph()
    G2.add_node('A', label='A')
    G2.add_node('B', label='B')
    G2.add_node('C', label='C')
    G2.add_node('D', label='D')
    G2.add_edge('A', 'B', label='a-b')
    G2.add_edge('B', 'C', label='b-c')
    G2.add_edge('A', 'D', label='a-d')
    assert graph_edit_distance(G1, G2, node_match=nmatch, edge_match=ematch) == 2