from itertools import chain, combinations
import pytest
import networkx as nx
def test_disjoin_cliques(self):
    G = nx.Graph(['ab', 'AB', 'AC', 'BC', '12', '13', '14', '23', '24', '34'])
    truth = {frozenset('ab'), frozenset('ABC'), frozenset('1234')}
    self._check_communities(G, truth)