from itertools import combinations
import pytest
import networkx as nx
@pytest.fixture()
def large_no_separating_set_graph():
    edge_list = [('A', 'B'), ('C', 'A'), ('C', 'B')]
    G = nx.DiGraph(edge_list)
    return G