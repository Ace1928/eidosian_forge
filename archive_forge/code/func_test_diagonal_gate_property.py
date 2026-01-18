import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_diagonal_gate_property():
    assert cirq.ThreeQubitDiagonalGate([2, 3, 5, 7, 0, 0, 0, 1]).diag_angles_radians == (2, 3, 5, 7, 0, 0, 0, 1)