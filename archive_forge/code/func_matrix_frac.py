from functools import wraps
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint
@wraps(MatrixFrac)
def matrix_frac(X, P):
    if isinstance(P, np.ndarray):
        invP = LA.inv(P)
        return QuadForm(X, (invP + np.conj(invP).T) / 2.0)
    else:
        return MatrixFrac(X, P)