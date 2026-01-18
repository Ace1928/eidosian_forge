import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def next_pow2(n):
    """ Return first power of 2 >= n.

    >>> next_pow2(3)
    4
    >>> next_pow2(8)
    8
    >>> next_pow2(0)
    1
    >>> next_pow2(1)
    1
    """
    if n <= 0:
        return 1
    n2 = 1 << int(n).bit_length()
    if n2 >> 1 == n:
        return n
    else:
        return n2