from __future__ import annotations
import math
from typing import Sequence
def gcd_float(numbers: Sequence[float], tol: float=1e-08) -> float:
    """
    Returns the greatest common divisor for a sequence of numbers.
    Uses a numerical tolerance, so can be used on floats

    Args:
        numbers: Sequence of numbers.
        tol: Numerical tolerance

    Returns:
        float: Greatest common divisor of numbers.
    """

    def pair_gcd_tol(a: float, b: float) -> float:
        """Calculate the Greatest Common Divisor of a and b.

        Unless b==0, the result will have the same sign as b (so that when
        b is divided by it, the result comes out positive).
        """
        while b > tol:
            a, b = (b, a % b)
        return a
    n = numbers[0]
    for i in numbers:
        n = pair_gcd_tol(n, i)
    return n