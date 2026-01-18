import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_condensation_mapping_and_members(self):
    G, C = self.gc[1]
    C = sorted(C, key=len, reverse=True)
    cG = nx.condensation(G)
    mapping = cG.graph['mapping']
    assert all((n in G for n in mapping))
    assert all((0 == cN for n, cN in mapping.items() if n in C[0]))
    assert all((1 == cN for n, cN in mapping.items() if n in C[1]))
    for n, d in cG.nodes(data=True):
        assert set(C[n]) == cG.nodes[n]['members']