import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.fixture
def multigraph_test_graph():
    G = nx.MultiGraph()
    G.add_edge(1, 2, weight=7)
    G.add_edge(1, 2, weight=70)
    return G