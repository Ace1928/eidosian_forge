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
def cases_test_fit_mle():
    skip_basic_fit = {'argus', 'foldnorm', 'truncpareto', 'truncweibull_min', 'ksone', 'levy_stable', 'studentized_range', 'kstwo', 'arcsine'}
    slow_basic_fit = {'alpha', 'betaprime', 'binom', 'bradford', 'burr12', 'chi', 'crystalball', 'dweibull', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'genexpon', 'genextreme', 'gennorm', 'genpareto', 'gompertz', 'halfgennorm', 'invgauss', 'invweibull', 'jf_skew_t', 'johnsonsb', 'johnsonsu', 'kappa3', 'kstwobign', 'loglaplace', 'lognorm', 'lomax', 'mielke', 'nakagami', 'nbinom', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powernorm', 'randint', 'rdist', 'recipinvgauss', 'rice', 't', 'uniform', 'weibull_max', 'wrapcauchy'}
    xslow_basic_fit = {'beta', 'betabinom', 'burr', 'exponweib', 'gausshyper', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'hypergeom', 'kappa4', 'loguniform', 'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius', 'nct', 'ncx2', 'nhypergeom', 'powerlognorm', 'reciprocal', 'rel_breitwigner', 'skellam', 'trapezoid', 'triang', 'truncnorm', 'tukeylambda', 'zipfian'}
    for dist in dict(distdiscrete + distcont):
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = 'tested separately'
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        elif dist in slow_basic_fit:
            reason = 'too slow (>= 0.25s)'
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        elif dist in xslow_basic_fit:
            reason = 'too slow (>= 1.0s)'
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        else:
            yield dist