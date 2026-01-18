import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_is_regular1(self):
    g = gen.cycle_graph(4)
    assert reg.is_regular(g)