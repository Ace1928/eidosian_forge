import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
def check_clifford_t(op, use_decomposition=False):
    """Checks whether the gate is in the standard Clifford+T basis.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, which is not a T-gate,
    this method checks that the transformation :math:`UPU^{\\dagger}` maps the Pauli tensor products
    :math:`P = {I, X, Y, Z}^{\\otimes N}` to Pauli tensor products using the decomposition of the
    matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be checked
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford+T or not.
    """
    if isinstance(op, Adjoint):
        base = op.base
    else:
        base = None
    if isinstance(op, _CLIFFORD_T_GATES) or isinstance(base, _CLIFFORD_T_GATES):
        return True
    if isinstance(op, _PARAMETER_GATES) or isinstance(base, _PARAMETER_GATES):
        theta = op.data[0]
        return False if qml.math.is_abstract(theta) else qml.math.allclose(qml.math.mod(theta, math.pi), 0.0)
    return _check_clifford_op(op, use_decomposition=use_decomposition)