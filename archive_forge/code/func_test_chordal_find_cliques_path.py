import pytest
import networkx as nx
def test_chordal_find_cliques_path(self):
    G = nx.path_graph(10)
    cliqueset = nx.chordal_graph_cliques(G)
    for u, v in G.edges():
        assert frozenset([u, v]) in cliqueset or frozenset([v, u]) in cliqueset