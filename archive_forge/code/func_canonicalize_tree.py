from cvxpy import problems
from cvxpy import settings as s
from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.constraints import (
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.lin_ops import lin_utils as lu
from cvxpy.reductions import InverseData, Solution
from cvxpy.reductions.complex2real.canonicalizers import CANON_METHODS as elim_cplx_methods
from cvxpy.reductions.reduction import Reduction
def canonicalize_tree(self, expr, real2imag, leaf_map):
    if type(expr) == cvxtypes.partial_problem():
        raise NotImplementedError()
    else:
        real_args = []
        imag_args = []
        for arg in expr.args:
            real_arg, imag_arg = self.canonicalize_tree(arg, real2imag, leaf_map)
            real_args.append(real_arg)
            imag_args.append(imag_arg)
        real_out, imag_out = self.canonicalize_expr(expr, real_args, imag_args, real2imag, leaf_map)
    return (real_out, imag_out)