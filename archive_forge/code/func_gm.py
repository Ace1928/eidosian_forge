import numbers
from collections import defaultdict
from fractions import Fraction
import numpy as np
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
def gm(t, x, y):
    length = t.size
    return SOC(t=reshape(x + y, (length,)), X=vstack([reshape(x - y, (1, length)), reshape(2 * t, (1, length))]), axis=0)