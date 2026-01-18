import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_expansion', (({cirq.X: 10, cirq.Y: -20}, {'X': 10, 'Y': -20}), ({cirq.Y: np.sqrt(0.5), cirq.H: 1}, {'X': np.sqrt(0.5), 'Y': np.sqrt(0.5), 'Z': np.sqrt(0.5)}), ({cirq.X: 2, cirq.H: 1}, {'X': 2 + np.sqrt(0.5), 'Z': np.sqrt(0.5)}), ({cirq.XX: -2, cirq.YY: 3j, cirq.ZZ: 4}, {'XX': -2, 'YY': 3j, 'ZZ': 4})))
def test_linear_combination_of_gates_has_correct_pauli_expansion(terms, expected_expansion):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12