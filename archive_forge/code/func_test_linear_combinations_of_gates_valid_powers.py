import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, exponent, expected_terms', (({cirq.X: 1}, 2, {cirq.I: 1}), ({cirq.X: 1}, 3, {cirq.X: 1}), ({cirq.Y: 0.5}, 10, {cirq.I: 2 ** (-10)}), ({cirq.Y: 0.5}, 11, {cirq.Y: 2 ** (-11)}), ({cirq.I: 1, cirq.X: 2, cirq.Y: 3, cirq.Z: 4}, 2, {cirq.I: 30, cirq.X: 4, cirq.Y: 6, cirq.Z: 8}), ({cirq.X: 1, cirq.Y: 1j}, 2, {}), ({cirq.X: 0.4, cirq.Y: 0.4}, 0, {cirq.I: 1})))
def test_linear_combinations_of_gates_valid_powers(terms, exponent, expected_terms):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_result = combination ** exponent
    expected_result = cirq.LinearCombinationOfGates(expected_terms)
    assert cirq.approx_eq(actual_result, expected_result)
    assert len(actual_result) == len(expected_terms)