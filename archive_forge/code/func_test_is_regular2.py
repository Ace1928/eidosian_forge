import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_is_regular2(self):
    g = gen.complete_graph(5)
    assert reg.is_regular(g)