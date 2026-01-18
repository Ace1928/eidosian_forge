import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('pauli', (cirq.X, cirq.Y, cirq.Z, cirq.I, 'I', 'X', 'Y', 'Z', 'i', 'x', 'y', 'z', 0, 1, 2, 3))
def test_mutable_pauli_string_dict_pauli_like(pauli):
    p = cirq.MutablePauliString()
    p[0] = pauli