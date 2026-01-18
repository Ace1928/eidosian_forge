import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
@staticmethod
def compute_matrix(phi):
    """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ControlledPhaseShift.matrix`

        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.ControlledPhaseShift.compute_matrix(torch.tensor(0.5))
            tensor([[1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 1.0+0.0j, 0.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 1.0+0.0j, 0.0000+0.0000j],
                    [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.8776+0.4794j]])
        """
    if qml.math.get_interface(phi) == 'tensorflow':
        p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
        if qml.math.ndim(p) == 0:
            return qml.math.diag([1, 1, 1, p])
        ones = qml.math.ones_like(p)
        diags = stack_last([ones, ones, ones, p])
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)
    signs = qml.math.array([0, 0, 0, 1], like=phi)
    arg = 1j * phi
    if qml.math.ndim(arg) == 0:
        return qml.math.diag(qml.math.exp(arg * signs))
    diags = qml.math.exp(qml.math.outer(arg, signs))
    return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(4, like=diags), diags)