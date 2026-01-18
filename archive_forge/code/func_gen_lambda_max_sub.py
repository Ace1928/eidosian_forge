from cvxpy import atoms
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
def gen_lambda_max_sub(expr, t):
    return [expr.args[0] == expr.args[0].T, expr.args[1] >> 0, t * expr.args[1] - expr.args[0] >> 0]