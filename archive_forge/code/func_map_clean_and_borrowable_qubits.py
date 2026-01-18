from typing import Optional
import cirq
@cirq._compat.deprecated(deadline='v1.4', fix='Use cirq.map_clean_and_borrowable_qubits instead.')
def map_clean_and_borrowable_qubits(circuit: cirq.AbstractCircuit, *, qm: Optional[cirq.QubitManager]=None) -> cirq.Circuit:
    """This method is deprecated. See docstring of `cirq.map_clean_and_borrowable_qubits`"""
    return cirq.map_clean_and_borrowable_qubits(circuit, qm=qm)