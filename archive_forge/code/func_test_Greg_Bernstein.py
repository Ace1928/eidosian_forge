import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_Greg_Bernstein():
    g1 = nx.Graph()
    g1.add_nodes_from(['N0', 'N1', 'N2', 'N3', 'N4'])
    g1.add_edge('N4', 'N1', weight=10.0, capacity=50, name='L5')
    g1.add_edge('N4', 'N0', weight=7.0, capacity=40, name='L4')
    g1.add_edge('N0', 'N1', weight=10.0, capacity=45, name='L1')
    g1.add_edge('N3', 'N0', weight=10.0, capacity=50, name='L0')
    g1.add_edge('N2', 'N3', weight=12.0, capacity=30, name='L2')
    g1.add_edge('N1', 'N2', weight=15.0, capacity=42, name='L3')
    solution = [['N1', 'N0', 'N3'], ['N1', 'N2', 'N3'], ['N1', 'N4', 'N0', 'N3']]
    result = list(nx.shortest_simple_paths(g1, 'N1', 'N3', weight='weight'))
    assert result == solution