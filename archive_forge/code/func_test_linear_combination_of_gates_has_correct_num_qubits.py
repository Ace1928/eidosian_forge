import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_num_qubits', (({cirq.X: 1}, 1), ({cirq.H: 10, cirq.S: -10j}, 1), ({cirq.XX: 1, cirq.YY: 2, cirq.ZZ: 3}, 2), ({cirq.CCZ: 0.1, cirq.CSWAP: 0.2}, 3)))
def test_linear_combination_of_gates_has_correct_num_qubits(terms, expected_num_qubits):
    combination = cirq.LinearCombinationOfGates(terms)
    assert combination.num_qubits() == expected_num_qubits