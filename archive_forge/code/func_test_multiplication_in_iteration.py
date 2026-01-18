import numpy as np
import pytest
import cirq
def test_multiplication_in_iteration():
    linear_dict = cirq.LinearDict({'u': 2, 'v': 1, 'w': -1})
    for v, c in linear_dict.items():
        if c > 0:
            linear_dict[v] *= 0
    assert linear_dict == cirq.LinearDict({'u': 0, 'v': 0, 'w': -1})
    assert linear_dict == cirq.LinearDict({'w': -1})