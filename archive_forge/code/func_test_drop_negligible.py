import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_drop_negligible():
    q0, = _make_qubits(1)
    sym = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 1e-10, cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** sym)
    expected = cirq.Circuit(cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z})) ** sym)
    circuit = cirq.drop_negligible_operations(circuit)
    circuit = cirq.drop_empty_moments(circuit)
    assert circuit == expected