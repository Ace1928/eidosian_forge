import pytest
import networkx as nx
from networkx.algorithms.connectivity.kcomponents import (
def test_davis_southern_women():
    G = nx.davis_southern_women_graph()
    result = nx.k_components(G)
    _check_connectivity(G, result)