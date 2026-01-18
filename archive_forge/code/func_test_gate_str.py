import numpy as np
import pytest
import sympy
import cirq
def test_gate_str():
    assert str(cirq.GlobalPhaseGate(1j)) == '1j'