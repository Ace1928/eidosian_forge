import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_geometric_edges_pos_attribute():
    G = nx.Graph()
    G.add_nodes_from([(0, {'position': (0, 0)}), (1, {'position': (0, 1)}), (2, {'position': (1, 0)})])
    expected_edges = [(0, 1), (0, 2)]
    assert expected_edges == nx.geometric_edges(G, radius=1, pos_name='position')