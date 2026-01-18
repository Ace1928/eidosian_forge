import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def is_weight(w) -> bool:
    """ Test if w is a valid weight vector.
        w must have nonnegative integer or fractional elements, and sum to 1.

    Examples
    --------
    >>> is_weight((Fraction(1,3), Fraction(2,3)))
    True
    >>> is_weight((Fraction(2,3), Fraction(2,3)))
    False
    >>> is_weight([.1, .9])
    False
    >>> import numpy as np
    >>> w = np.array([.1, .9])
    >>> is_weight(w)
    False
    >>> w = np.array([0, 0, 1])
    >>> is_weight(w)
    True
    >>> w = (0,1,0)
    >>> is_weight(w)
    True

    """
    if isinstance(w, np.ndarray):
        w = w.tolist()
    valid_elems = all((v >= 0 and isinstance(v, (numbers.Integral, Fraction)) for v in w))
    return valid_elems and sum(w) == 1