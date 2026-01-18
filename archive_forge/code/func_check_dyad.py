import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def check_dyad(w, w_dyad):
    """Check that w_dyad is a valid dyadic completion of w.

    Parameters
    ----------
    w : Sequence
        Tuple of nonnegative fractional or integer weights that sum to 1.
    w_dyad : Sequence
        Proposed dyadic completion of w.

    Returns
    -------
    bool
        True if w_dyad is a valid dyadic completion of w.


    Examples
    --------
    >>> w = (Fraction(1,3), Fraction(1,3), Fraction(1,3))
    >>> w_dyad =(Fraction(1,4), Fraction(1,4), Fraction(1,4), Fraction(1,4))
    >>> check_dyad(w, w_dyad)
    True

    If the weight vector is already dyadic, it is its own completion.

    >>> w = (Fraction(1,4), 0, Fraction(3,4))
    >>> check_dyad(w, w)
    True

    Integer input should also be accepted

    >>> w = (1, 0, 0)
    >>> check_dyad(w, w)
    True

    w is not a valid weight vector here because it doesn't sum to 1

    >>> w = (Fraction(2,3), 1)
    >>> check_dyad(w, w)
    False

    w_dyad isn't the correct dyadic completion.

    >>> w = (Fraction(2,5), Fraction(3,5))
    >>> w_dyad = (Fraction(3,8), Fraction(4,8), Fraction(1,8))
    >>> check_dyad(w, w_dyad)
    False

    The correct dyadic completion.

    >>> w = (Fraction(2,5), Fraction(3,5))
    >>> w_dyad = (Fraction(2,8), Fraction(3,8), Fraction(3,8))
    >>> check_dyad(w, w_dyad)
    True

    """
    if not (is_weight(w) and is_dyad_weight(w_dyad)):
        return False
    if w == w_dyad:
        return True
    if len(w_dyad) == len(w) + 1:
        return w == tuple((Fraction(v, 1 - w_dyad[-1]) for v in w_dyad[:-1]))
    else:
        return False