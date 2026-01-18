import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_graph_with_reserved_keywords(self):
    G = nx.Graph()
    G = self.build_graph(G)
    G.nodes['E']['n'] = 'keyword'
    G.edges['A', 'B']['u'] = 'keyword'
    G.edges['A', 'B']['v'] = 'keyword'
    A = nx.nx_agraph.to_agraph(G)