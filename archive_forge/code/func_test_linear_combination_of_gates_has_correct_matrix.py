import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_matrix', (({cirq.I: 2, cirq.X: 3, cirq.Y: 4, cirq.Z: 5j}, np.array([[2 + 5j, 3 - 4j], [3 + 4j, 2 - 5j]])), ({cirq.XX: 0.5, cirq.YY: -0.5}, np.rot90(np.diag([1, 0, 0, 1]))), ({cirq.CCZ: 3j}, np.diag([3j, 3j, 3j, 3j, 3j, 3j, 3j, -3j]))))
def test_linear_combination_of_gates_has_correct_matrix(terms, expected_matrix):
    combination = cirq.LinearCombinationOfGates(terms)
    assert np.all(combination.matrix() == expected_matrix)