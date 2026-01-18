import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_cannot_multiply_by_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.X(q) * cirq.Z(q) ** 0.5
    with pytest.raises(TypeError):
        _ = cirq.Z(q) ** 0.5 * cirq.X(q)
    with pytest.raises(TypeError):
        _ = cirq.Y(q) * cirq.S(q)