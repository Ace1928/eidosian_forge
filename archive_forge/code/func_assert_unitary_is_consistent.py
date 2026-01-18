from typing import Any
import cirq
import numpy as np
def assert_unitary_is_consistent(val: Any, ignoring_global_phase: bool=False):
    if not isinstance(val, (cirq.Operation, cirq.Gate)):
        return
    if not cirq.has_unitary(val):
        return
    u = cirq.unitary(val)
    assert not (u is None or u is NotImplemented)
    assert cirq.is_unitary(u)
    if isinstance(val, cirq.Operation):
        qubits = val.qubits
        decomposition = cirq.decompose_once(val, default=None)
    else:
        qubits = tuple(cirq.LineQid.for_gate(val))
        decomposition = cirq.decompose_once_with_qubits(val, qubits, default=None)
    if decomposition is None or decomposition is NotImplemented:
        return
    c = cirq.Circuit(decomposition)
    if len(c.all_qubits().difference(qubits)) == 0:
        return
    clean_qubits = tuple((q for q in c.all_qubits() if isinstance(q, cirq.ops.CleanQubit)))
    borrowable_qubits = tuple((q for q in c.all_qubits() if isinstance(q, cirq.ops.BorrowableQubit)))
    qubit_order = clean_qubits + borrowable_qubits + qubits
    assert set(qubit_order) == c.all_qubits()
    qid_shape = cirq.qid_shape(qubit_order)
    full_unitary = cirq.apply_unitaries(decomposition, qubits=qubit_order, args=cirq.ApplyUnitaryArgs.for_unitary(qid_shape=qid_shape), default=None)
    if full_unitary is None:
        raise ValueError(f'apply_unitaries failed on the decomposition of {val}')
    vol = np.prod(qid_shape, dtype=np.int64)
    full_unitary = full_unitary.reshape((vol, vol))
    vol = np.prod(cirq.qid_shape(borrowable_qubits + qubits), dtype=np.int64)
    clean_qubits_zero_subspace = full_unitary[:vol, :vol]
    expected = np.kron(np.eye(2 ** len(borrowable_qubits), dtype=np.complex128), u)
    if ignoring_global_phase:
        cirq.testing.assert_allclose_up_to_global_phase(clean_qubits_zero_subspace, expected, atol=1e-08)
    else:
        np.testing.assert_allclose(clean_qubits_zero_subspace, expected, atol=1e-08)