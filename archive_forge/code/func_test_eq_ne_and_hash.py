import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_eq_ne_and_hash():
    eq = cirq.testing.EqualsTester()
    for pauli0, invert0, pauli1, invert1, e in itertools.product(_paulis, _bools, _paulis, _bools, (0.125, -0.25, 1)):
        eq.add_equality_group(cirq.PauliInteractionGate(pauli0, invert0, pauli1, invert1, exponent=e))