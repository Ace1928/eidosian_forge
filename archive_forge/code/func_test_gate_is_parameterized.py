import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_gate_is_parameterized():
    gate = cirq.PauliStringPhasorGate(dps_empty)
    assert not cirq.is_parameterized(gate)
    assert not cirq.is_parameterized(gate ** 0.1)
    assert cirq.is_parameterized(gate ** sympy.Symbol('a'))