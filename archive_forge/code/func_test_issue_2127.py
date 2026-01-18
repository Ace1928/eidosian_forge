import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
def test_issue_2127(self):
    """Test from issue 2127"""
    G = nx.DiGraph()
    G.add_edge('A', 'C')
    G.add_edge('A', 'B')
    G.add_edge('C', 'E')
    G.add_edge('C', 'D')
    G.add_edge('E', 'G')
    G.add_edge('E', 'F')
    G.add_edge('G', 'I')
    G.add_edge('G', 'H')
    tc = nx.transitive_closure(G)
    btc = nx.Graph()
    for v in tc.nodes():
        btc.add_node((0, v))
        btc.add_node((1, v))
    for u, v in tc.edges():
        btc.add_edge((0, u), (1, v))
    top_nodes = {n for n in btc if n[0] == 0}
    matching = hopcroft_karp_matching(btc, top_nodes)
    vertex_cover = to_vertex_cover(btc, matching, top_nodes)
    independent_set = set(G) - {v for _, v in vertex_cover}
    assert {'B', 'D', 'F', 'I', 'H'} == independent_set