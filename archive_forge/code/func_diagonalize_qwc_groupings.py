from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def diagonalize_qwc_groupings(qwc_groupings):
    """Diagonalizes a list of qubit-wise commutative groupings of Pauli strings.

    Args:
        qwc_groupings (list[list[Observable]]): a list of mutually qubit-wise commutative groupings
            of Pauli string observables

    Returns:
        tuple:

            * list[list[Operation]]: a list of instances of the qwc_rotation
              template which diagonalizes the qubit-wise commuting grouping,
              order corresponding to qwc_groupings
            * list[list[Observable]]: a list of QWC groupings diagonalized in the
              computational basis, order corresponding to qwc_groupings

    **Example**

    >>> qwc_group_1 = [qml.X(0) @ qml.Z(1),
                       qml.X(0) @ qml.Y(3),
                       qml.Z(1) @ qml.Y(3)]
    >>> qwc_group_2 = [qml.Y(0),
                       qml.Y(0) @ qml.X(2),
                       qml.X(1) @ qml.Z(3)]
    >>> post_rotations, diag_groupings = diagonalize_qwc_groupings([qwc_group_1, qwc_group_2])
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [RX(1.5707963267948966, wires=[0]),
      RY(-1.5707963267948966, wires=[2]),
      RY(-1.5707963267948966, wires=[1])]]
    >>> diag_groupings
    [[Z(0) @ Z(1),
     Z(0) @ Z(3),
     Z(1) @ Z(3)],
    [Z(0),
     Z(0) @ Z(2),
     Z(1) @ Z(3)]]
    """
    post_rotations = []
    diag_groupings = []
    m_groupings = len(qwc_groupings)
    for i in range(m_groupings):
        diagonalizing_unitary, diag_grouping = diagonalize_qwc_pauli_words(qwc_groupings[i])
        post_rotations.append(diagonalizing_unitary)
        diag_groupings.append(diag_grouping)
    return (post_rotations, diag_groupings)