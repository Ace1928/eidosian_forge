from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def ctrl_decomp_bisect(target_operation: Operator, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    Automatically selects the best algorithm based on the matrix (uses specialized more efficient
    algorithms if the matrix has a certain form, otherwise falls back to the general algorithm).
    These algorithms are defined in section 3.1 and 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example:**

    >>> op = qml.T(0) # uses OD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†─┤
    1: ─├●────────│──────────├●────────│──────────┤
    2: ─├●────────│──────────├●────────│──────────┤
    3: ─╰●────────│──────────╰●────────│──────────┤
    4: ───────────├●───────────────────├●─────────┤
    5: ───────────╰●───────────────────╰●─────────┤
    >>> op = qml.QubitUnitary([[0,1j],[1j,0]], 0) # uses MD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──H─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†──H─┤
    1: ────├●────────│──────────├●────────│─────────────┤
    2: ────├●────────│──────────├●────────│─────────────┤
    3: ────╰●────────│──────────╰●────────│─────────────┤
    4: ──────────────├●───────────────────├●────────────┤
    5: ──────────────╰●───────────────────╰●────────────┤
    >>> op = qml.Hadamard(0) # uses general algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──U(M0)─╭X──U(M1)†──U(M2)─╭X──U(M2)†─╭X──U(M2)─╭X──U(M2)†─╭X──U(M1)─╭X──U(M0)─┤
    1: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    2: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    3: ────────│─────────────────│──────────╰●────────│──────────╰●────────│─────────┤
    4: ────────├●────────────────├●───────────────────├●───────────────────├●────────┤
    5: ────────╰●────────────────╰●───────────────────╰●───────────────────╰●────────┤

    """
    if len(target_operation.wires) > 1:
        raise ValueError(f'The target operation must be a single-qubit operation, instead got {target_operation}.')
    target_matrix = target_operation.matrix()
    target_wire = target_operation.wires
    target_matrix = _convert_to_su2(target_matrix)
    target_matrix_imag = np.imag(target_matrix)
    if np.isclose(target_matrix_imag[1, 0], 0) and np.isclose(target_matrix_imag[0, 1], 0):
        return _ctrl_decomp_bisect_od(target_matrix, target_wire, control_wires)
    if np.isclose(target_matrix_imag[0, 0], 0) and np.isclose(target_matrix_imag[1, 1], 0):
        return _ctrl_decomp_bisect_md(target_matrix, target_wire, control_wires)
    return _ctrl_decomp_bisect_general(target_matrix, target_wire, control_wires)