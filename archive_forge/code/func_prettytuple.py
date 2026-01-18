import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def prettytuple(t):
    """ Use the string representation of objects in a tuple.
    """
    return '(' + ', '.join((str(f) for f in t)) + ')'