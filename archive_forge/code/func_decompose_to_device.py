from typing import Sequence
from typing import Union
import cirq
from cirq_ionq import ionq_gateset
@cirq._compat.deprecated(deadline='v0.16', fix='Use cirq.optimize_for_target_gateset(circuit, gateset=cirq_ionq.IonQTargetGateset(atol)) instead.')
def decompose_to_device(operation: cirq.Operation, atol: float=1e-08) -> cirq.OP_TREE:
    """Decompose operation to ionq native operations.


    Merges single qubit operations and decomposes two qubit operations
    into CZ gates.

    Args:
        operation: `cirq.Operation` to decompose.
        atol: absolute error tolerance to use when declaring two unitary
            operations equal.

    Returns:
        cirq.OP_TREE containing decomposed operations.

    Raises:
        ValueError: If supplied operation cannot be decomposed
            for the ionq device.

    """
    return cirq.optimize_for_target_gateset(cirq.Circuit(operation), gateset=ionq_gateset.IonQTargetGateset(), ignore_failures=False).all_operations()