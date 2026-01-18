import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
def rz_matrix(phi: float) -> np.ndarray:
    """
    Computes an RZ rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RZ rotation matrix.
    """
    return RZGate(phi).to_matrix()