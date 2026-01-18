import numpy as np
from scipy import sparse
from cvxpy.atoms.suppfunc import SuppFuncAtom
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CONVEX_ATTRIBUTES
def conic_repr_of_set(self):
    return (self._A, self._b, self._K_sels)