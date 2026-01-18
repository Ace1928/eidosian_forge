import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_create_using_keyword_arguments(self):
    pytest.raises(nx.NetworkXError, gn_graph, 100, create_using=Graph())
    pytest.raises(nx.NetworkXError, gnr_graph, 100, 0.5, create_using=Graph())
    pytest.raises(nx.NetworkXError, gnc_graph, 100, create_using=Graph())
    G = gn_graph(100, seed=1)
    MG = gn_graph(100, create_using=MultiDiGraph(), seed=1)
    assert sorted(G.edges()) == sorted(MG.edges())
    G = gnr_graph(100, 0.5, seed=1)
    MG = gnr_graph(100, 0.5, create_using=MultiDiGraph(), seed=1)
    assert sorted(G.edges()) == sorted(MG.edges())
    G = gnc_graph(100, seed=1)
    MG = gnc_graph(100, create_using=MultiDiGraph(), seed=1)
    assert sorted(G.edges()) == sorted(MG.edges())
    G = scale_free_graph(100, alpha=0.3, beta=0.4, gamma=0.3, delta_in=0.3, delta_out=0.1, initial_graph=nx.cycle_graph(4, create_using=MultiDiGraph), seed=1)
    pytest.raises(ValueError, scale_free_graph, 100, 0.5, 0.4, 0.3)
    pytest.raises(ValueError, scale_free_graph, 100, alpha=-0.3)
    pytest.raises(ValueError, scale_free_graph, 100, beta=-0.3)
    pytest.raises(ValueError, scale_free_graph, 100, gamma=-0.3)