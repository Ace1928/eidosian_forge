import warnings
from typing import Any, List, Sequence, Optional
import numpy as np
from cirq import devices, linalg, ops, protocols
from cirq.testing import lin_alg_utils
def assert_qasm_is_consistent_with_unitary(val: Any):
    """Uses `val._unitary_` to check `val._qasm_`'s behavior."""
    try:
        import qiskit
    except ImportError:
        warnings.warn("Skipped assert_qasm_is_consistent_with_unitary because qiskit isn't installed to verify against.")
        return
    unitary = protocols.unitary(val, None)
    if unitary is None:
        return
    if isinstance(val, ops.Operation):
        qubits: Sequence[ops.Qid] = val.qubits
        op = val
    elif isinstance(val, ops.Gate):
        qid_shape = protocols.qid_shape(val)
        remaining_shape = list(qid_shape)
        controls = getattr(val, 'control_qubits', None)
        if controls is not None:
            for i, q in zip(reversed(range(len(controls))), reversed(controls)):
                if q is not None:
                    remaining_shape.pop(i)
        qubits = devices.LineQid.for_qid_shape(remaining_shape)
        op = val.on(*qubits)
    else:
        raise NotImplementedError(f"Don't know how to test {val!r}")
    args = protocols.QasmArgs(qubit_id_map={q: f'q[{i}]' for i, q in enumerate(qubits)})
    qasm = protocols.qasm(op, args=args, default=None)
    if qasm is None:
        return
    num_qubits = len(qubits)
    header = f'\nOPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{num_qubits}];\n'
    qasm = header + qasm
    qasm_unitary = None
    try:
        result = qiskit.execute(qiskit.QuantumCircuit.from_qasm_str(qasm), backend=qiskit.Aer.get_backend('unitary_simulator'))
        qasm_unitary = result.result().get_unitary()
        qasm_unitary = _reorder_indices_of_matrix(qasm_unitary, list(reversed(range(num_qubits))))
        lin_alg_utils.assert_allclose_up_to_global_phase(qasm_unitary, unitary, rtol=1e-08, atol=1e-08)
    except Exception as ex:
        p_unitary: Optional[np.ndarray]
        p_qasm_unitary: Optional[np.ndarray]
        if qasm_unitary is not None:
            p_unitary, p_qasm_unitary = linalg.match_global_phase(unitary, qasm_unitary)
        else:
            p_unitary = None
            p_qasm_unitary = None
        raise AssertionError(f'QASM not consistent with cirq.unitary(op) up to global phase.\n\nop:\n{_indent(repr(op))}\n\ncirq.unitary(op):\n{_indent(repr(unitary))}\n\nGenerated QASM:\n\n{_indent(qasm)}\n\nUnitary of generated QASM:\n{_indent(repr(qasm_unitary))}\n\nPhased matched cirq.unitary(op):\n{_indent(repr(p_unitary))}\n\nPhased matched unitary of generated QASM:\n{_indent(repr(p_qasm_unitary))}\n\nUnderlying error:\n{_indent(str(ex))}')