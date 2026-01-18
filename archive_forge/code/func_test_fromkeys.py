import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('keys, coefficient, terms_expected', (((), 10, {}), (('X',), 2, {'X': 2}), (('a', 'b', 'c', 'd'), 0.5, {'a': 0.5, 'b': 0.5, 'c': 0.5, 'd': 0.5}), (('b', 'c', 'd', 'e'), -2j, {'b': -2j, 'c': -2j, 'd': -2j, 'e': -2j})))
def test_fromkeys(keys, coefficient, terms_expected):
    actual = cirq.LinearDict.fromkeys(keys, coefficient)
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual