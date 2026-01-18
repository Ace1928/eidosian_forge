import numbers
import pytest
import networkx as nx
from ..generators import (
def test_alternating_havel_hakimi_graph(self):
    aseq = []
    bseq = []
    G = alternating_havel_hakimi_graph(aseq, bseq)
    assert len(G) == 0
    aseq = [0, 0]
    bseq = [0, 0]
    G = alternating_havel_hakimi_graph(aseq, bseq)
    assert len(G) == 4
    assert G.number_of_edges() == 0
    aseq = [3, 3, 3, 3]
    bseq = [2, 2, 2, 2, 2]
    pytest.raises(nx.NetworkXError, alternating_havel_hakimi_graph, aseq, bseq)
    bseq = [2, 2, 2, 2, 2, 2]
    G = alternating_havel_hakimi_graph(aseq, bseq)
    assert sorted((d for n, d in G.degree())) == [2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    aseq = [2, 2, 2, 2, 2, 2]
    bseq = [3, 3, 3, 3]
    G = alternating_havel_hakimi_graph(aseq, bseq)
    assert sorted((d for n, d in G.degree())) == [2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    aseq = [2, 2, 2, 1, 1, 1]
    bseq = [3, 3, 3]
    G = alternating_havel_hakimi_graph(aseq, bseq)
    assert G.is_multigraph()
    assert not G.is_directed()
    assert sorted((d for n, d in G.degree())) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    GU = nx.projected_graph(nx.Graph(G), range(len(aseq)))
    assert GU.number_of_nodes() == 6
    GD = nx.projected_graph(nx.Graph(G), range(len(aseq), len(aseq) + len(bseq)))
    assert GD.number_of_nodes() == 3
    G = reverse_havel_hakimi_graph(aseq, bseq, create_using=nx.Graph)
    assert not G.is_multigraph()
    assert not G.is_directed()
    pytest.raises(nx.NetworkXError, alternating_havel_hakimi_graph, aseq, bseq, create_using=nx.DiGraph)
    pytest.raises(nx.NetworkXError, alternating_havel_hakimi_graph, aseq, bseq, create_using=nx.DiGraph)
    pytest.raises(nx.NetworkXError, alternating_havel_hakimi_graph, aseq, bseq, create_using=nx.MultiDiGraph)