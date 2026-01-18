from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def get_knots_bsplines(x=None, df=None, knots=None, degree=3, spacing='quantile', lower_bound=None, upper_bound=None, all_knots=None):
    """knots for use in B-splines

    There are two main options for the knot placement

    - quantile spacing with multiplicity of boundary knots
    - equal spacing extended to boundary or exterior knots

    The first corresponds to splines as used by patsy. the second is the
    knot spacing for P-Splines.
    """
    if all_knots is not None:
        return all_knots
    x_min = x.min()
    x_max = x.max()
    if degree < 0:
        raise ValueError('degree must be greater than 0 (not %r)' % (degree,))
    if int(degree) != degree:
        raise ValueError('degree must be an integer (not %r)' % (degree,))
    if df is None and knots is None:
        raise ValueError('must specify either df or knots')
    order = degree + 1
    if df is not None:
        n_inner_knots = df - order
        if n_inner_knots < 0:
            raise ValueError('df=%r is too small for degree=%r; must be >= %s' % (df, degree, df - n_inner_knots))
        if knots is not None:
            if len(knots) != n_inner_knots:
                raise ValueError('df=%s with degree=%r implies %s knots, but %s knots were provided' % (df, degree, n_inner_knots, len(knots)))
        elif spacing == 'quantile':
            knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            inner_knots = _R_compat_quantile(x, knot_quantiles)
        elif spacing == 'equal':
            grid = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            inner_knots = x_min + grid * (x_max - x_min)
            diff_knots = inner_knots[1] - inner_knots[0]
        else:
            raise ValueError('incorrect option for spacing')
    if knots is not None:
        inner_knots = knots
    if lower_bound is None:
        lower_bound = np.min(x)
    if upper_bound is None:
        upper_bound = np.max(x)
    if lower_bound > upper_bound:
        raise ValueError('lower_bound > upper_bound (%r > %r)' % (lower_bound, upper_bound))
    inner_knots = np.asarray(inner_knots)
    if inner_knots.ndim > 1:
        raise ValueError('knots must be 1 dimensional')
    if np.any(inner_knots < lower_bound):
        raise ValueError('some knot values (%s) fall below lower bound (%r)' % (inner_knots[inner_knots < lower_bound], lower_bound))
    if np.any(inner_knots > upper_bound):
        raise ValueError('some knot values (%s) fall above upper bound (%r)' % (inner_knots[inner_knots > upper_bound], upper_bound))
    if spacing == 'equal':
        diffs = np.arange(1, order + 1) * diff_knots
        lower_knots = inner_knots[0] - diffs[::-1]
        upper_knots = inner_knots[-1] + diffs
        all_knots = np.concatenate((lower_knots, inner_knots, upper_knots))
    else:
        all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))
    all_knots.sort()
    return all_knots