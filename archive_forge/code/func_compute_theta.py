import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def compute_theta(alpha):
    """Maps the angles alpha of the multi-controlled rotations decomposition of a uniformly controlled rotation
     to the rotation angles used in the Gray code implementation.

    Args:
        alpha (tensor_like): alpha parameters

    Returns:
        (tensor_like): rotation angles theta
    """
    ln = alpha.shape[-1]
    k = np.log2(ln)
    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
        for j in range(len(M_trans[0])):
            M_trans[i, j] = _matrix_M_entry(j, i)
    theta = qml.math.transpose(qml.math.dot(M_trans, qml.math.transpose(alpha)))
    return theta / 2 ** k