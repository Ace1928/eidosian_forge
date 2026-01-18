from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import devices, ops, protocols
def compute_cphase_exponents_for_fsim_decomposition(fsim_gate: 'cirq.FSimGate') -> Sequence[Tuple[float, float]]:
    """Returns intervals of CZPowGate exponents valid for FSim decomposition.

    Ideal intervals associated with the constraints are closed, but due to
    numerical error the caller should not assume the endpoints themselves
    are valid for the decomposition. See `decompose_cphase_into_two_fsim`
    for details on how FSimGate parameters constrain the phase angle of
    CZPowGate.

    Args:
        fsim_gate: FSimGate into which CZPowGate would be decomposed.

    Returns:
        Sequence of 2-tuples each consisting of the minimum and maximum
        value of the exponent for which CZPowGate can be decomposed into
        two FSimGates. The intervals are cropped to [0, 2]. The function
        returns zero, one or two intervals.

    Raises:
        ValueError: if the fsim_gate contains symbolic parameters.
    """

    def nonempty_intervals(intervals: Sequence[Tuple[float, float]]) -> Sequence[Tuple[float, float]]:
        return tuple(((a, b) for a, b in intervals if a < b))
    if isinstance(fsim_gate.theta, sympy.Expr) or isinstance(fsim_gate.phi, sympy.Expr):
        raise ValueError('Symbolic arguments not supported')
    bound1 = abs(_asinsin(fsim_gate.theta))
    bound2 = abs(_asinsin(fsim_gate.phi / 2))
    min_exponent_1 = 4 * min(bound1, bound2) / np.pi
    max_exponent_1 = 4 * max(bound1, bound2) / np.pi
    assert min_exponent_1 <= max_exponent_1
    min_exponent_2 = 2 - max_exponent_1
    max_exponent_2 = 2 - min_exponent_1
    assert min_exponent_2 <= max_exponent_2
    if max_exponent_1 < min_exponent_2:
        return nonempty_intervals([(min_exponent_1, max_exponent_1), (min_exponent_2, max_exponent_2)])
    if max_exponent_2 < min_exponent_1:
        return nonempty_intervals([(min_exponent_2, max_exponent_2), (min_exponent_1, max_exponent_1)])
    min_exponent = min(min_exponent_1, min_exponent_2)
    max_exponent = max(max_exponent_1, max_exponent_2)
    return nonempty_intervals([(min_exponent, max_exponent)])