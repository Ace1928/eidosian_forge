import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms', ({cirq.X(q0): -2, cirq.H(q0): 2}, {cirq.X(q0): -2, cirq.H(q1): 2j}, {cirq.X(q0): 1, cirq.CZ(q0, q1): 3}, {cirq.X(q0): 1 + 1j, cirq.CZ(q1, q2): 0.5}))
def test_linear_combination_of_operations_is_consistent(terms):
    combination_1 = cirq.LinearCombinationOfOperations(terms)
    combination_2 = cirq.LinearCombinationOfOperations({})
    combination_2.update(terms)
    combination_3 = cirq.LinearCombinationOfOperations({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient
    assert combination_1 == combination_2 == combination_3