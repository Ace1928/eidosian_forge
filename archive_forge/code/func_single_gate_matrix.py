from __future__ import annotations
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError
def single_gate_matrix(gate: str, params: list[float] | None=None) -> np.ndarray:
    """Get the matrix for a single qubit.

    Args:
        gate: the single qubit gate name
        params: the operation parameters op['params']
    Returns:
        array: A numpy array representing the matrix
    Raises:
        QiskitError: If a gate outside the supported set is passed in for the
            ``Gate`` argument.
    """
    if params is None:
        params = []
    if gate == 'U':
        gc = gates.UGate
    elif gate == 'u3':
        gc = gates.U3Gate
    elif gate == 'h':
        gc = gates.HGate
    elif gate == 'u':
        gc = gates.UGate
    elif gate == 'p':
        gc = gates.PhaseGate
    elif gate == 'u2':
        gc = gates.U2Gate
    elif gate == 'u1':
        gc = gates.U1Gate
    elif gate == 'rz':
        gc = gates.RZGate
    elif gate == 'id':
        gc = gates.IGate
    elif gate == 'sx':
        gc = gates.SXGate
    elif gate == 'x':
        gc = gates.XGate
    else:
        raise QiskitError('Gate is not a valid basis gate for this simulator: %s' % gate)
    return gc(*params).to_matrix()