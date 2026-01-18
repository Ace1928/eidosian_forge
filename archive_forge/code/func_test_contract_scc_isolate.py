import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_contract_scc_isolate(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    assert list(cG.nodes()) == [0]
    assert list(cG.edges()) == []