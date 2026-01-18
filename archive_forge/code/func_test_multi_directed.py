import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_multi_directed(self):
    self.agraph_checks(nx.MultiDiGraph())