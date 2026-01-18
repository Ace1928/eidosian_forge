import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import string
@pytest.mark.parametrize('kmax', [5, 10])
def test_binary_tree(kmax):
    n = 2 ** kmax
    elements = get_elements(n)
    dis = DisjointSet(elements)
    rng = np.random.RandomState(seed=0)
    for k in 2 ** np.arange(kmax):
        for i in range(0, n, 2 * k):
            r1, r2 = rng.randint(0, k, size=2)
            a, b = (elements[i + r1], elements[i + k + r2])
            assert not dis.connected(a, b)
            assert dis.merge(a, b)
            assert dis.connected(a, b)
        assert elements == list(dis)
        roots = [dis[i] for i in elements]
        expected_indices = np.arange(n) - np.arange(n) % (2 * k)
        expected = [elements[i] for i in expected_indices]
        assert roots == expected