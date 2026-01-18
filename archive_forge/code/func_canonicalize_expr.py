import numpy as np
from cvxpy import settings
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dgp2dcp.canonicalizers import DgpCanonMethods
def canonicalize_expr(self, expr, args):
    if type(expr) in self.canon_methods:
        return self.canon_methods[type(expr)](expr, args)
    else:
        return (expr.copy(args), [])