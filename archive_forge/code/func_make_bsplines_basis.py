from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def make_bsplines_basis(x, df, degree):
    """ make a spline basis for x """
    all_knots, _, _, _ = compute_all_knots(x, df, degree)
    basis, der_basis, der2_basis = _eval_bspline_basis(x, all_knots, degree)
    return (basis, der_basis, der2_basis)