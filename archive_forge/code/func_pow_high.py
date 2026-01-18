import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def pow_high(p, max_denom: int=1024):
    """ Return (t,1,x) power tuple

        x <= t^(1/p) 1^(1-1/p)

        user wants the epigraph variable t
    """
    assert p > 1
    p = Fraction(1 / Fraction(p)).limit_denominator(max_denom)
    if 1 / p == int(1 / p):
        return (int(1 / p), (p, 1 - p))
    return (1 / p, (p, 1 - p))