import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('gate', (cirq.X, cirq.Y, cirq.XX, cirq.CZ, cirq.CSWAP, cirq.FREDKIN))
def test_empty_linear_combination_of_gates_accepts_all_gates(gate):
    combination = cirq.LinearCombinationOfGates({})
    combination[gate] = -0.5j
    assert len(combination) == 1