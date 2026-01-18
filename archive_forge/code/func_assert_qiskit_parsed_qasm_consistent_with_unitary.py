import warnings
from typing import Any, List, Sequence, Optional
import numpy as np
from cirq import devices, linalg, ops, protocols
from cirq.testing import lin_alg_utils
def assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, unitary):
    try:
        import qiskit
    except ImportError:
        return
    num_qubits = int(np.log2(len(unitary)))
    result = qiskit.execute(qiskit.QuantumCircuit.from_qasm_str(qasm), backend=qiskit.Aer.get_backend('unitary_simulator'))
    qiskit_unitary = result.result().get_unitary()
    qiskit_unitary = _reorder_indices_of_matrix(qiskit_unitary, list(reversed(range(num_qubits))))
    lin_alg_utils.assert_allclose_up_to_global_phase(unitary, qiskit_unitary, rtol=1e-08, atol=1e-08)