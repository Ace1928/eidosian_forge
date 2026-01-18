import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_map_qubits():
    q0, q1, q2, q3, q4, q5 = _make_qubits(6)
    qubit_map = {q1: q2, q0: q3}
    before = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y}), exponent_neg=0.1)
    after = cirq.PauliStringPhasor(cirq.PauliString({q3: cirq.Z, q2: cirq.Y}), exponent_neg=0.1)
    assert before.map_qubits(qubit_map) == after
    qubit_map = {q1: q3, q0: q4, q2: q5}
    before = cirq.PauliStringPhasor(cirq.PauliString({q0: cirq.Z, q1: cirq.Y}), qubits=[q0, q1, q2], exponent_neg=0.1)
    after = cirq.PauliStringPhasor(cirq.PauliString({q4: cirq.Z, q3: cirq.Y}), qubits=[q4, q3, q5], exponent_neg=0.1)
    assert before.map_qubits(qubit_map) == after