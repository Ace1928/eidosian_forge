from sympy.combinatorics.subsets import Subset, ksubsets
from sympy.testing.pytest import raises
def test_ksubsets():
    assert list(ksubsets([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]
    assert list(ksubsets([1, 2, 3, 4, 5], 2)) == [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]