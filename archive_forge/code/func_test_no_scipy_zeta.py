import pytest
import networkx as nx
def test_no_scipy_zeta():
    zeta2 = 1.6449340668482264
    assert abs(zeta2 - nx.generators.community._hurwitz_zeta(2, 1, 0.0001)) < 0.01