import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('gates', ((cirq.X, cirq.T, cirq.T, cirq.X, cirq.Z), (cirq.CZ, cirq.XX, cirq.YY, cirq.ZZ), (cirq.TOFFOLI, cirq.TOFFOLI, cirq.FREDKIN)))
def test_in_place_manipulations_of_linear_combination_of_gates(gates):
    a = cirq.LinearCombinationOfGates({})
    b = cirq.LinearCombinationOfGates({})
    for i, gate in enumerate(gates):
        a += gate
        b -= gate
        prefix = gates[:i + 1]
        expected_a = cirq.LinearCombinationOfGates(collections.Counter(prefix))
        expected_b = -expected_a
        assert_linear_combinations_are_equal(a, expected_a)
        assert_linear_combinations_are_equal(b, expected_b)