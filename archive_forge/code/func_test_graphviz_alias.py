import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_graphviz_alias(self):
    G = self.build_graph(nx.Graph())
    pos_graphviz = nx.nx_agraph.graphviz_layout(G)
    pos_pygraphviz = nx.nx_agraph.pygraphviz_layout(G)
    assert pos_graphviz == pos_pygraphviz