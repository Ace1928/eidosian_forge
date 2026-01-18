from cvxpy import atoms
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
def ratio_sub(expr, t):
    x, y = expr.args
    if y.is_nonneg():
        return [x <= t * y]
    elif y.is_nonpos():
        return [x >= t * y]
    else:
        raise ValueError("The denominator's sign must be known.")