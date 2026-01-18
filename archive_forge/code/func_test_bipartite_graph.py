from itertools import chain, combinations
import pytest
import networkx as nx
def test_bipartite_graph(self):
    G = nx.complete_bipartite_graph(self.N // 2, self.N // 2)
    truth = {frozenset(G)}
    self._check_communities(G, truth)