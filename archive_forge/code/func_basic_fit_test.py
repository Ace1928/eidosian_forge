import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def basic_fit_test(self, dist_name, method):
    N = 5000
    dist_data = dict(distcont + distdiscrete)
    rng = np.random.default_rng(self.seed)
    dist = getattr(stats, dist_name)
    shapes = np.array(dist_data[dist_name])
    bounds = np.empty((len(shapes) + 2, 2), dtype=np.float64)
    bounds[:-2, 0] = shapes / 10.0 ** np.sign(shapes)
    bounds[:-2, 1] = shapes * 10.0 ** np.sign(shapes)
    bounds[-2] = (0, 10)
    bounds[-1] = (1e-16, 10)
    loc = rng.uniform(*bounds[-2])
    scale = rng.uniform(*bounds[-1])
    ref = list(dist_data[dist_name]) + [loc, scale]
    if getattr(dist, 'pmf', False):
        ref = ref[:-1]
        ref[-1] = np.floor(loc)
        data = dist.rvs(*ref, size=N, random_state=rng)
        bounds = bounds[:-1]
    if getattr(dist, 'pdf', False):
        data = dist.rvs(*ref, size=N, random_state=rng)
    with npt.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'overflow encountered')
        res = stats.fit(dist, data, bounds, method=method, optimizer=self.opt)
    nlff_names = {'mle': 'nnlf', 'mse': '_penalized_nlpsf'}
    nlff_name = nlff_names[method]
    assert_nlff_less_or_close(dist, data, res.params, ref, **self.tols, nlff_name=nlff_name)