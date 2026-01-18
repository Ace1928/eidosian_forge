from typing import Any
import numpy as np
from cirq import devices, protocols, ops, circuits
from cirq.testing import lin_alg_utils
def assert_decompose_is_consistent_with_unitary(val: Any, ignoring_global_phase: bool=False):
    """Uses `val._unitary_` to check `val._phase_by_`'s behavior."""
    __tracebackhide__ = True
    expected = protocols.unitary(val, None)
    if expected is None:
        return
    if isinstance(val, ops.Operation):
        qubits = val.qubits
        dec = protocols.decompose_once(val, default=None)
    else:
        qubits = tuple(devices.LineQid.for_gate(val))
        dec = protocols.decompose_once_with_qubits(val, qubits, default=None)
    if dec is None:
        return
    c = circuits.Circuit(dec)
    if len(c.all_qubits().difference(qubits)):
        ancilla = tuple(c.all_qubits().difference(qubits))
        qubit_order = ancilla + qubits
        actual = c.unitary(qubit_order=qubit_order)
        qid_shape = protocols.qid_shape(qubits)
        vol = np.prod(qid_shape, dtype=np.int64)
        actual = actual[:vol, :vol]
    else:
        actual = c.unitary(qubit_order=qubits)
    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-08)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-08)