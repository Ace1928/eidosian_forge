from typing import Dict, Tuple
import numpy as np
from cirq import value
from cirq._doc import document
def pow_pauli_combination(ai: value.Scalar, ax: value.Scalar, ay: value.Scalar, az: value.Scalar, exponent: int) -> Tuple[value.Scalar, value.Scalar, value.Scalar, value.Scalar]:
    """Computes non-negative integer power of single-qubit Pauli combination.

    Returns scalar coefficients bi, bx, by, bz such that

        bi I + bx X + by Y + bz Z = (ai I + ax X + ay Y + az Z)^exponent

    Correctness of the formulas below follows from the binomial expansion
    and the fact that for any real or complex vector (ax, ay, az) and any
    non-negative integer k:

         [ax X + ay Y + az Z]^(2k) = (ax^2 + ay^2 + az^2)^k I

    """
    if exponent == 0:
        return (1, 0, 0, 0)
    v = np.sqrt(ax * ax + ay * ay + az * az).item()
    s = (ai + v) ** exponent
    t = (ai - v) ** exponent
    ci = (s + t) / 2
    if s == t:
        cxyz = exponent * ai ** (exponent - 1)
    else:
        cxyz = (s - t) / 2
        cxyz = cxyz / v
    return (ci, cxyz * ax, cxyz * ay, cxyz * az)