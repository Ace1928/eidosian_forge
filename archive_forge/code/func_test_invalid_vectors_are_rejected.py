import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms, valid_vectors, invalid_vectors', (({'X': 2}, 'X', ('A', 'B')), ({'X': 2, 'Y': -2}, ('X', 'Y', 'Z'), ('A', 'B'))))
def test_invalid_vectors_are_rejected(terms, valid_vectors, invalid_vectors):
    linear_dict = cirq.LinearDict(terms, validator=lambda v: v in valid_vectors)
    with pytest.raises(ValueError):
        linear_dict += cirq.LinearDict.fromkeys(invalid_vectors, 1)
    assert linear_dict == cirq.LinearDict(terms)
    for vector in invalid_vectors:
        with pytest.raises(ValueError):
            linear_dict[vector] += 1
    assert linear_dict == cirq.LinearDict(terms)
    with pytest.raises(ValueError):
        linear_dict.update(cirq.LinearDict.fromkeys(invalid_vectors, 1))
    assert linear_dict == cirq.LinearDict(terms)