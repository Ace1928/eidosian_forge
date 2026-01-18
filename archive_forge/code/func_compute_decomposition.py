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
def compute_decomposition(phi, wires):
    """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.ControlledPhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.ControlledPhaseShift.compute_decomposition(1.234, wires=(0,1))
        [PhaseShift(0.617, wires=[0]),
         CNOT(wires=[0, 1]),
         PhaseShift(-0.617, wires=[1]),
         CNOT(wires=[0, 1]),
         PhaseShift(0.617, wires=[1])]

        """
    return [qml.PhaseShift(phi / 2, wires=wires[0]), qml.CNOT(wires=wires), qml.PhaseShift(-phi / 2, wires=wires[1]), qml.CNOT(wires=wires), qml.PhaseShift(phi / 2, wires=wires[1])]