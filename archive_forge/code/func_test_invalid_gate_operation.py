import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_invalid_gate_operation():
    three_qubit_gate = cirq.testing.ThreeQubitGate()
    single_qubit = [cirq.GridQubit(0, 0)]
    with pytest.raises(ValueError, match='number of qubits'):
        cirq.GateOperation(three_qubit_gate, single_qubit)