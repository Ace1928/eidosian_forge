from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
def test_consistent():
    assert consistent([A], [A])
    assert consistent([B], [B])
    assert not consistent([A], [C])
    assert consistent([A, B], [A, B])
    assert consistent([B, A], [A, B])
    assert not consistent([B, A], [B])
    assert not consistent([B, A], [B, C])