import pytest
import numpy as np
import cirq
from cirq.testing import assert_allclose_up_to_global_phase
def test_clifford_decompose_one_qubit():
    """Two random instance for one qubit decomposition."""
    qubits = cirq.LineQubit.range(1)
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=1), qubits=qubits, prng=np.random.RandomState())
    cirq.act_on(cirq.X, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.X(qubits[0]), cirq.H(qubits[0]), cirq.S(qubits[0]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-07)
    qubits = cirq.LineQubit.range(1)
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=1), qubits=qubits, prng=np.random.RandomState())
    cirq.act_on(cirq.Z, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.X, args, qubits=[qubits[0]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.Z(qubits[0]), cirq.H(qubits[0]), cirq.S(qubits[0]), cirq.H(qubits[0]), cirq.X(qubits[0]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-07)