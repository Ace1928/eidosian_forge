import numpy as np
import pytest
import cirq
def test_addition_in_iteration():
    linear_dict = cirq.LinearDict({'a': 2, 'b': 1, 'c': 0, 'd': -1, 'e': -2})
    for v in linear_dict:
        linear_dict[v] += 1
    assert linear_dict == cirq.LinearDict({'a': 3, 'b': 2, 'c': 0, 'd': 0, 'e': -1})
    assert linear_dict == cirq.LinearDict({'a': 3, 'b': 2, 'e': -1})