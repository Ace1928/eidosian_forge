import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def single_qubit_matrix_to_phxz(mat: np.ndarray, atol: float=0) -> Optional[ops.PhasedXZGate]:
    """Implements a single-qubit operation with a PhasedXZ gate.

    Under the hood, this uses deconstruct_single_qubit_matrix_into_angles which
    converts the given matrix to a series of three rotations around the Z, Y, Z
    axes. This is then converted to a phased X rotation followed by a Z, in the
    form of a single PhasedXZ gate.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        atol: A limit on the amount of error introduced by the
            construction.

    Returns:
        A PhasedXZ gate that implements the given matrix, or None if it is
        close to identity (trace distance <= atol).
    """
    xy_turn, xy_phase_turn, total_z_turn = _deconstruct_single_qubit_matrix_into_gate_turns(mat)
    g = ops.PhasedXZGate(axis_phase_exponent=2 * xy_phase_turn, x_exponent=2 * xy_turn, z_exponent=2 * total_z_turn)
    if protocols.trace_distance_bound(g) <= atol:
        return None
    if math.isclose(abs(xy_turn), 0.5, abs_tol=atol):
        g = ops.PhasedXZGate(axis_phase_exponent=2 * xy_phase_turn + total_z_turn, x_exponent=1, z_exponent=0)
    return g