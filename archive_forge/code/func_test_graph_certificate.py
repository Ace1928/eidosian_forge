from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
def test_graph_certificate():
    import random

    def randomize_graph(size, g):
        p = list(range(size))
        random.shuffle(p)
        g1a = {}
        for k, v in g1.items():
            g1a[p[k]] = [p[i] for i in v]
        return g1a
    g1 = {0: [2, 3, 7], 1: [4, 5, 7], 2: [0, 4, 6], 3: [0, 6, 7], 4: [1, 2, 5], 5: [1, 4, 6], 6: [2, 3, 5], 7: [0, 1, 3]}
    g2 = {0: [2, 3, 7], 1: [2, 4, 5], 2: [0, 1, 5], 3: [0, 6, 7], 4: [1, 5, 6], 5: [1, 2, 4], 6: [3, 4, 7], 7: [0, 3, 6]}
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(8, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a
    g1 = {0: [8, 1, 9, 7], 1: [0, 9, 3, 4], 2: [3, 4, 6, 7], 3: [1, 2, 5, 6], 4: [8, 1, 2, 5], 5: [9, 3, 4, 7], 6: [8, 2, 3, 7], 7: [0, 2, 5, 6], 8: [0, 9, 4, 6], 9: [8, 0, 5, 1]}
    g2 = {0: [1, 2, 5, 6], 1: [0, 9, 5, 7], 2: [0, 4, 6, 7], 3: [8, 9, 6, 7], 4: [8, 2, 6, 7], 5: [0, 9, 8, 1], 6: [0, 2, 3, 4], 7: [1, 2, 3, 4], 8: [9, 3, 4, 5], 9: [8, 1, 3, 5]}
    c1 = graph_certificate(g1)
    c2 = graph_certificate(g2)
    assert c1 != c2
    g1a = randomize_graph(10, g1)
    c1a = graph_certificate(g1a)
    assert c1 == c1a