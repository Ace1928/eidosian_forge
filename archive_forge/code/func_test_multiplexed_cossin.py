import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
def test_multiplexed_cossin():
    angle_1 = np.random.random_sample() * 2 * np.pi
    angle_2 = np.random.random_sample() * 2 * np.pi
    c1, s1 = (np.cos(angle_1), np.sin(angle_1))
    c2, s2 = (np.cos(angle_2), np.sin(angle_2))
    multiplexed_ry = [[c1, 0, -s1, 0], [0, c2, 0, -s2], [s1, 0, c1, 0], [0, s2, 0, c2]]
    multiplexed_ry = np.array(multiplexed_ry)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(2)]
    circuit = cirq.Circuit(_multiplexed_cossin(qubits, [angle_1, angle_2]))
    assert cirq.approx_eq(multiplexed_ry, circuit.unitary(), atol=1e-09)
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all((isinstance(op.gate, gates) for op in circuit.all_operations()))