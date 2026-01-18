import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_finding_routines(self):
    G = nx.Graph({1: [2], 2: [3], 3: [4], 4: [5], 5: [6]})
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 7)
    G.add_edge(3, 6)
    G.add_edge(4, 6)
    assert nxt.find_alternating_4_cycle(G) == [1, 2, 3, 6]
    TG = nxt.find_threshold_graph(G)
    assert nxt.is_threshold_graph(TG)
    assert sorted(TG.nodes()) == [1, 2, 3, 4, 5, 7]
    cs = nxt.creation_sequence(dict(TG.degree()), with_labels=True)
    assert nxt.find_creation_sequence(G) == cs