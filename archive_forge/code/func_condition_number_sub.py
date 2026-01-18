from cvxpy import atoms
from cvxpy.atoms.affine import binary_operators as bin_op
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
def condition_number_sub(expr, t):
    A = expr.args[0]
    n = A.shape[0]
    u = Variable(pos=True)
    prom_ut = promote(u * t, (n,))
    prom_u = promote(u, (n,))
    tmp_expr1 = A - diag_vec(prom_u)
    tmp_expr2 = diag_vec(prom_ut) - A
    return [upper_tri(A) == upper_tri(A.T), PSD(A), PSD(tmp_expr1), PSD(tmp_expr2)]