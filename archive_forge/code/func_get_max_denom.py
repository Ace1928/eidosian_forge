import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def get_max_denom(tup):
    """ Get the maximum denominator in a sequence of ``Fraction`` and ``int`` objects
    """
    return max((Fraction(f).denominator for f in tup))