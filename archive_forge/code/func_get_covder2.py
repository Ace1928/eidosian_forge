from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def get_covder2(smoother, k_points=3, integration_points=None, skip_ctransf=False, deriv=2):
    """
    Approximate integral of cross product of second derivative of smoother

    This uses scipy.integrate simps to compute an approximation to the
    integral of the smoother derivative cross-product at knots plus k_points
    in between knots.
    """
    try:
        from scipy.integrate import simpson
    except ImportError:
        from scipy.integrate import simps as simpson
    knots = smoother.knots
    if integration_points is None:
        x = _get_integration_points(knots, k_points=k_points)
    else:
        x = integration_points
    d2 = smoother.transform(x, deriv=deriv, skip_ctransf=skip_ctransf)
    covd2 = simpson(d2[:, :, None] * d2[:, None, :], x=x, axis=0)
    return covd2