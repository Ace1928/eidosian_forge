from sympy.multipledispatch.conflict import (supercedes, ordering, ambiguities,
def test_ambiguities():
    signatures = [[A], [B], [A, B], [B, A], [A, C]]
    expected = {((A, B), (B, A))}
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))
    signatures = [[A], [B], [A, B], [B, A], [A, C], [B, B]]
    expected = set()
    result = ambiguities(signatures)
    assert set(map(frozenset, expected)) == set(map(frozenset, result))