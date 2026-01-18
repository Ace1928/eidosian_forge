import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
def test_k_factor_trivial(self):
    g = gen.cycle_graph(4)
    f = reg.k_factor(g, 2)
    assert g.edges == f.edges