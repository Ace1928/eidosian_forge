from collections import defaultdict
import pytest
import networkx as nx
from networkx.algorithms.communicability_alg import communicability, communicability_exp
def test_communicability(self):
    answer = {0: {0: 1.5430806348152435, 1: 1.1752011936438012}, 1: {0: 1.1752011936438012, 1: 1.5430806348152435}}
    result = communicability(nx.path_graph(2))
    for k1, val in result.items():
        for k2 in val:
            assert answer[k1][k2] == pytest.approx(result[k1][k2], abs=1e-07)