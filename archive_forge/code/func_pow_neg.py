import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def pow_neg(p, max_denom: int=1024):
    """ Return (x,t,1) power tuple

        1 <= x^(p/(p-1)) t^(-1/(p-1))

        user wants the epigraph variable t
    """
    assert p < 0
    p = Fraction(p)
    p = Fraction(p / (p - 1)).limit_denominator(max_denom)
    return (p / (p - 1), (p, 1 - p))