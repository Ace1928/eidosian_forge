import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_multi_clifford_decompose_by_unitary():
    n, num_ops = (5, 20)
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for _ in range(10):
        qubits = cirq.LineQubit.range(n)
        ops = []
        for _ in range(num_ops):
            g = np.random.randint(len(gate_candidate))
            indices = (np.random.randint(n),) if g < 5 else np.random.choice(n, 2, replace=False)
            ops.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        gate = cirq.CliffordGate.from_op_list(ops, qubits)
        decomposed_ops = cirq.decompose(gate.on(*qubits))
        circ = cirq.Circuit(decomposed_ops)
        circ.append(cirq.I.on_each(qubits))
        cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(gate), cirq.unitary(circ), atol=1e-07)