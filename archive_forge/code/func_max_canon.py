import numpy as np
from cvxpy.atoms import promote, reshape
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
def max_canon(expr, args):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    t = Variable(shape)
    if axis is None:
        promoted_t = promote(t, x.shape)
    elif axis == 0:
        promoted_t = Constant(np.ones((x.shape[0], 1))) @ reshape(t, (1, x.shape[1]))
    else:
        promoted_t = reshape(t, (x.shape[0], 1)) @ Constant(np.ones((1, x.shape[1])))
    constraints = [x <= promoted_t]
    return (t, constraints)