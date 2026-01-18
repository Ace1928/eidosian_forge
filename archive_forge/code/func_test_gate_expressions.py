import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('expression, expected_result', (((cirq.X + cirq.Z) / np.sqrt(2), cirq.H), (cirq.X - cirq.Y, -cirq.Y + cirq.X), (cirq.X + cirq.S - cirq.X, cirq.S), (cirq.Y - 2 * cirq.Y, -cirq.Y), (cirq.rx(0.2), np.cos(0.1) * cirq.I - 1j * np.sin(0.1) * cirq.X), (1j * cirq.H * 1j, -cirq.H), (-1j * cirq.Y, cirq.ry(np.pi)), (np.sqrt(-1j) * cirq.S, cirq.rz(np.pi / 2)), (0.5 * (cirq.IdentityGate(2) + cirq.XX + cirq.YY + cirq.ZZ), cirq.SWAP), ((cirq.IdentityGate(2) + 1j * (cirq.XX + cirq.YY) + cirq.ZZ) / 2, cirq.ISWAP), (cirq.CNOT + 0 * cirq.SWAP, cirq.CNOT), (0.5 * cirq.FREDKIN, cirq.FREDKIN / 2), (cirq.FREDKIN * 0.5, cirq.FREDKIN / 2), (((cirq.X + cirq.Y) / np.sqrt(2)) ** 2, cirq.I), ((cirq.X + cirq.Z) ** 3, 2 * (cirq.X + cirq.Z)), ((cirq.X + 1j * cirq.Y) ** 2, cirq.LinearCombinationOfGates({})), ((cirq.X - 1j * cirq.Y) ** 2, cirq.LinearCombinationOfGates({})), (((3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13) ** 24, cirq.I), (((3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13) ** 25, (3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13), ((cirq.X + cirq.Y + cirq.Z) ** 0, cirq.I), ((cirq.X - 1j * cirq.Y) ** 0, cirq.I)))
def test_gate_expressions(expression, expected_result):
    assert_linear_combinations_are_equal(expression, expected_result)