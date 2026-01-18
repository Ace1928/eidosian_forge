from __future__ import division
import warnings
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import is_sparse
from cvxpy.utilities.linalg import sparse_cholesky
def quad_form(x, P, assume_PSD: bool=False):
    """ Alias for :math:`x^T P x`.

    Parameters
    ----------
    x : vector argument.
    P : matrix argument.
    assume_PSD : P is assumed to be PSD without checking.
    """
    x, P = map(Expression.cast_to_const, (x, P))
    if not P.ndim == 2 or P.shape[0] != P.shape[1] or max(x.shape, (1,))[0] != P.shape[0]:
        raise Exception('Invalid dimensions for arguments.')
    if x.is_constant():
        return x.H @ P @ x
    elif P.is_constant():
        if assume_PSD:
            P = psd_wrap(P)
        return QuadForm(x, P)
    else:
        raise Exception('At least one argument to quad_form must be non-variable.')