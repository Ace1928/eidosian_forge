import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_qubits', (({}, ()), ({cirq.I(q0): 1, cirq.H(q0): 0.001j}, (q0,)), ({cirq.X(q0): 1j, cirq.H(q1): 2j}, (q0, q1)), ({cirq.Y(q0): -1, cirq.CZ(q0, q1): 3000.0}, (q0, q1)), ({cirq.Z(q0): -1j, cirq.CNOT(q1, q2): 0.25}, (q0, q1, q2))))
def test_linear_combination_of_operations_has_correct_qubits(terms, expected_qubits):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert combination.qubits == expected_qubits