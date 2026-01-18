import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_GraphMLWriter_add_graphs(self):
    gmlw = GraphMLWriter()
    G = self.simple_directed_graph
    H = G.copy()
    gmlw.add_graphs([G, H])