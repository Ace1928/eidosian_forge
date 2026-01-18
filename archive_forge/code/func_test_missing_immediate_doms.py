import pytest
import networkx as nx
def test_missing_immediate_doms(self):
    g = nx.DiGraph()
    edges = [('entry_1', 'b1'), ('b1', 'b2'), ('b2', 'b3'), ('b3', 'exit'), ('entry_2', 'b3')]
    g.add_edges_from(edges)
    nx.dominance_frontiers(g, 'entry_1')