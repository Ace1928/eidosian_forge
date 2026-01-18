import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import string
@pytest.mark.parametrize('direction', ['forwards', 'backwards'])
@pytest.mark.parametrize('n', [10, 100])
def test_linear_union_sequence(n, direction):
    elements = get_elements(n)
    dis = DisjointSet(elements)
    assert elements == list(dis)
    indices = list(range(n - 1))
    if direction == 'backwards':
        indices = indices[::-1]
    for it, i in enumerate(indices):
        assert not dis.connected(elements[i], elements[i + 1])
        assert dis.merge(elements[i], elements[i + 1])
        assert dis.connected(elements[i], elements[i + 1])
        assert dis.n_subsets == n - 1 - it
    roots = [dis[i] for i in elements]
    if direction == 'forwards':
        assert all((elements[0] == r for r in roots))
    else:
        assert all((elements[-2] == r for r in roots))
    assert not dis.merge(elements[0], elements[-1])