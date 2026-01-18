import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_degree_histogram_empty():
    G = nx.Graph()
    assert nx.degree_histogram(G) == []