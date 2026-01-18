import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def make_frac(a, denom):
    """ Approximate ``a/sum(a)`` with tuple of fractions with denominator *exactly* ``denom``.

    >>> a = [.123, .345, .532]
    >>> make_frac(a,10)
    (Fraction(1, 10), Fraction(2, 5), Fraction(1, 2))
    >>> make_frac(a,100)
    (Fraction(3, 25), Fraction(7, 20), Fraction(53, 100))
    >>> make_frac(a,1000)
    (Fraction(123, 1000), Fraction(69, 200), Fraction(133, 250))
    """
    a = np.array(a, dtype=float) / sum(a)
    b = (denom * a).astype(int)
    err = b / float(denom) - a
    inds = np.argsort(err)[:denom - sum(b)]
    b[inds] += 1
    denom = int(denom)
    b = b.tolist()
    return tuple((Fraction(v, denom) for v in b))