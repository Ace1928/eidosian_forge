import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def single_qubit_matrix_to_pauli_rotations(mat: np.ndarray, atol: float=0) -> List[Tuple[ops.Pauli, float]]:
    """Implements a single-qubit operation with few rotations.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        atol: A limit on the amount of absolute error introduced by the
            construction.

    Returns:
        A list of (Pauli, half_turns) tuples that, when applied in order,
        perform the desired operation.
    """

    def is_clifford_rotation(half_turns):
        return near_zero_mod(half_turns, 0.5, atol=atol)

    def to_quarter_turns(half_turns):
        return round(2 * half_turns) % 4

    def is_quarter_turn(half_turns):
        return is_clifford_rotation(half_turns) and to_quarter_turns(half_turns) % 2 == 1

    def is_half_turn(half_turns):
        return is_clifford_rotation(half_turns) and to_quarter_turns(half_turns) == 2

    def is_no_turn(half_turns):
        return is_clifford_rotation(half_turns) and to_quarter_turns(half_turns) == 0
    z_rad_before, y_rad, z_rad_after = linalg.deconstruct_single_qubit_matrix_into_angles(mat)
    z_ht_before = z_rad_before / np.pi - 0.5
    m_ht = y_rad / np.pi
    m_pauli: ops.Pauli = ops.X
    z_ht_after = z_rad_after / np.pi + 0.5
    if is_clifford_rotation(z_ht_before):
        if (is_quarter_turn(z_ht_before) or is_quarter_turn(z_ht_after)) ^ (is_half_turn(m_ht) and is_no_turn(z_ht_before - z_ht_after)):
            z_ht_before += 0.5
            z_ht_after -= 0.5
            m_pauli = ops.Y
        if is_half_turn(z_ht_before) or is_half_turn(z_ht_after):
            z_ht_before -= 1
            z_ht_after += 1
            m_ht = -m_ht
    if is_no_turn(m_ht):
        z_ht_before += z_ht_after
        z_ht_after = 0
    elif is_half_turn(m_ht):
        z_ht_after -= z_ht_before
        z_ht_before = 0
    rotation_list = [(ops.Z, z_ht_before), (m_pauli, m_ht), (ops.Z, z_ht_after)]
    return [(pauli, ht) for pauli, ht in rotation_list if not is_no_turn(ht)]