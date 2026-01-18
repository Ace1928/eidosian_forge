from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
def lambda_max_canon(expr, args):
    A = args[0]
    n = A.shape[0]
    t = Variable()
    prom_t = promote(t, (n,))
    tmp_expr = diag_vec(prom_t) - A
    constr = [PSD(tmp_expr)]
    if not A.is_symmetric():
        ut = upper_tri(A)
        lt = upper_tri(A.T)
        constr.append(ut == lt)
    return (t, constr)