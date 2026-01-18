from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture()
def large_collider_graph():
    edge_list = [('A', 'B'), ('C', 'B'), ('B', 'D'), ('D', 'E'), ('B', 'F'), ('G', 'E')]
    G = nx.DiGraph(edge_list)
    return G