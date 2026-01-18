import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def single_qubit_matrix_to_phased_x_z(mat: np.ndarray, atol: float=0) -> List[ops.Gate]:
    """Implements a single-qubit operation with a PhasedX and Z gate.

    If one of the gates isn't needed, it will be omitted.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        atol: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of gates that, when applied in order, perform the desired
            operation.
    """
    xy_turn, xy_phase_turn, total_z_turn = _deconstruct_single_qubit_matrix_into_gate_turns(mat)
    result = [ops.PhasedXPowGate(exponent=2 * xy_turn, phase_exponent=2 * xy_phase_turn), ops.Z ** (2 * total_z_turn)]
    result = [g for g in result if protocols.trace_distance_bound(g) > atol]
    if len(result) == 2 and math.isclose(abs(xy_turn), 0.5, abs_tol=atol):
        return [ops.PhasedXPowGate(phase_exponent=2 * xy_phase_turn + total_z_turn)]
    return result