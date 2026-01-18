import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_map_qubits_raises():
    q = cirq.LineQubit.range(3)
    pauli_string = cirq.X(q[0]) * cirq.Y(q[1]) * cirq.Z(q[2])
    with pytest.raises(ValueError, match='must have a key for every qubit'):
        pauli_string.map_qubits({q[0]: q[1]})