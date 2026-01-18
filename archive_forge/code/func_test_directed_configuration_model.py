import pytest
import networkx as nx
def test_directed_configuration_model():
    G = nx.directed_configuration_model([], [], seed=0)
    assert len(G) == 0