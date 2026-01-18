import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def dyad_completion(w):
    """ Return the dyadic completion of ``w``.

        Return ``w`` if ``w`` is already dyadic.

        We assume the input is a tuple of nonnegative Fractions or integers which sum to 1.

    Examples
    --------
    >>> w = (Fraction(1,3), Fraction(1,3), Fraction(1, 3))
    >>> dyad_completion(w)
    (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4))

    >>> w = (Fraction(1,3), Fraction(1,5), Fraction(7, 15))
    >>> dyad_completion(w)
    (Fraction(5, 16), Fraction(3, 16), Fraction(7, 16), Fraction(1, 16))

    >>> w = (1, 0, 0.0, Fraction(0,1))
    >>> dyad_completion(w)
    (Fraction(1, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))
    """
    w = tuple((Fraction(v) for v in w))
    non_dyad_dens = [v.denominator for v in w if not is_power2(v.denominator)]
    if len(non_dyad_dens) > 0:
        d = max(non_dyad_dens)
        p = next_pow2(d)
        w_aug = tuple((Fraction(v * d, p) for v in w)) + (Fraction(p - d, p),)
        return dyad_completion(w_aug)
    else:
        return w