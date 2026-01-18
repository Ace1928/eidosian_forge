import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms_1, terms_2', (({}, {'X': 1e-09}), ({'X': 1e-12}, {'X': 0}), ({'X': 5e-10}, {'Y': 2e-11}), ({'X': 1.000000001}, {'X': 1, 'Z': 0})))
def test_approximately_equal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert cirq.approx_eq(linear_dict_1, linear_dict_2)
    assert cirq.approx_eq(linear_dict_2, linear_dict_1)