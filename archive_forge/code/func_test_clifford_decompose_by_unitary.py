import pytest
import numpy as np
import cirq
from cirq.testing import assert_allclose_up_to_global_phase
def test_clifford_decompose_by_unitary():
    """Validate the decomposition of random Clifford Tableau by unitary matrix.

    Due to the exponential growth in dimension, it cannot validate very large number of qubits.
    """
    n, num_ops = (5, 20)
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for seed in range(100):
        prng = np.random.RandomState(seed)
        t = cirq.CliffordTableau(num_qubits=n)
        qubits = cirq.LineQubit.range(n)
        expect_circ = cirq.Circuit()
        args = cirq.CliffordTableauSimulationState(tableau=t, qubits=qubits, prng=prng)
        for _ in range(num_ops):
            g = prng.randint(len(gate_candidate))
            indices = (prng.randint(n),) if g < 5 else prng.choice(n, 2, replace=False)
            cirq.act_on(gate_candidate[g], args, qubits=[qubits[i] for i in indices], allow_decompose=False)
            expect_circ.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
        circ = cirq.Circuit(ops)
        circ.append(cirq.I.on_each(qubits))
        expect_circ.append(cirq.I.on_each(qubits))
        assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-07)