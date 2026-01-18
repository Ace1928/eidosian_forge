import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_add_number_paulisum():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    psum = cirq.PauliSum.from_pauli_strings([pstr1]) + 1.3
    assert psum == cirq.PauliSum.from_pauli_strings([pstr1, cirq.PauliString({}, 1.3)])