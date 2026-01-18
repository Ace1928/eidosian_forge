import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
def test_empty_scipy(self):
    G = nx.Graph()
    assert _pagerank_scipy(G) == {}