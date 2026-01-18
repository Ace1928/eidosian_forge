import abc
from typing import Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
@staticmethod
def elemwise_grad_to_diag(value, rows, cols):
    """Converts elementwise gradient into a diagonal matrix for Atom._grad()

        Args:
            value: A scalar or NumPy matrix.

        Returns:
            A SciPy CSC sparse matrix.
        """
    if not np.isscalar(value):
        value = value.ravel(order='F')
    return sp.dia_matrix((np.atleast_1d(value), [0]), shape=(rows, cols)).tocsc()