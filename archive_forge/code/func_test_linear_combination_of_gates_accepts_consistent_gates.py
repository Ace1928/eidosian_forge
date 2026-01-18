import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms', ({cirq.X: -2, cirq.H: 2}, {cirq.XX: 1, cirq.YY: 1j, cirq.ZZ: -1}, {cirq.TOFFOLI: 0.5j, cirq.FREDKIN: 0.5}))
def test_linear_combination_of_gates_accepts_consistent_gates(terms):
    combination_1 = cirq.LinearCombinationOfGates(terms)
    combination_2 = cirq.LinearCombinationOfGates({})
    combination_2.update(terms)
    combination_3 = cirq.LinearCombinationOfGates({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient
    assert combination_1 == combination_2 == combination_3