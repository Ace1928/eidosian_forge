import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_and(self):
    ev = self.eview(self.G)
    some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
    if self.G.is_directed():
        assert ev & some_edges == {(0, 1, 0)}
        assert some_edges & ev == {(0, 1, 0)}
    else:
        assert ev & some_edges == {(0, 1, 0), (1, 0, 0)}
        assert some_edges & ev == {(0, 1, 0), (1, 0, 0)}