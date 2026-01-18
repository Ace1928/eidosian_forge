import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def testGraph1(self):
    G1 = getCanonical()
    G2 = nx.Graph()
    G2.add_node('A', label='A')
    G2.add_node('B', label='B')
    G2.add_node('D', label='D')
    G2.add_node('E', label='E')
    G2.add_edge('A', 'B', label='a-b')
    G2.add_edge('B', 'D', label='b-d')
    G2.add_edge('D', 'E', label='d-e')
    assert graph_edit_distance(G1, G2, node_match=nmatch, edge_match=ematch) == 3