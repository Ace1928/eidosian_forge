import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def is_dyad_weight(w) -> bool:
    """ Test if a vector is a valid dyadic weight vector.

        w must be nonnegative, sum to 1, and have integer or dyadic fractional elements.

        Examples
        --------
        >>> is_dyad_weight((Fraction(1,2), Fraction(1,2)))
        True
        >>> is_dyad_weight((Fraction(1,3), Fraction(2,3)))
        False
        >>> is_dyad_weight((0, 1, 0))
        True
    """
    return is_weight(w) and all((is_dyad(f) for f in w))