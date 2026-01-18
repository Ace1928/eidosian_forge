from typing import AbstractSet, Any, Dict, Union
import numpy as np
import sympy
import cirq
from cirq import value, _compat
from cirq.ops import raw_types
def qft(*qubits: 'cirq.Qid', without_reverse: bool=False, inverse: bool=False) -> 'cirq.Operation':
    """The quantum Fourier transform.

    Transforms a qubit register from the computational basis to the frequency
    basis.

    The inverse quantum Fourier transform is `cirq.qft(*qubits)**-1` or
    equivalently `cirq.inverse(cirq.qft(*qubits))`.

    Args:
        *qubits: The qubits to apply the qft to.
        without_reverse: When set, swap gates at the end of the qft are omitted.
            This reverses the qubit order relative to the standard qft effect,
            but makes the gate cheaper to apply.
        inverse: If set, the inverse qft is performed instead of the qft.
            Equivalent to calling `cirq.inverse` on the result, or raising it
            to the -1.

    Returns:
        A `cirq.Operation` applying the qft to the given qubits.
    """
    result = QuantumFourierTransformGate(len(qubits), without_reverse=without_reverse).on(*qubits)
    if inverse:
        result = cirq.inverse(result)
    return result