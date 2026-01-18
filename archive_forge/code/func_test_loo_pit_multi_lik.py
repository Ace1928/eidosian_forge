from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import (
from scipy.special import logsumexp
from scipy.stats import linregress, norm, halfcauchy
from xarray import DataArray, Dataset
from xarray_einstats.stats import XrContinuousRV
from ...data import concat, convert_to_inference_data, from_dict, load_arviz_data
from ...rcparams import rcParams
from ...stats import (
from ...stats.stats import _gpinv
from ...stats.stats_utils import get_log_likelihood
from ..helpers import check_multiple_attrs, multidim_models  # pylint: disable=unused-import
def test_loo_pit_multi_lik():
    rng = np.random.default_rng(0)
    post_pred = rng.standard_normal(size=(4, 100, 10))
    obs = np.quantile(post_pred, np.linspace(0, 1, 10))
    obs[0] *= 0.9
    obs[-1] *= 1.1
    idata = from_dict(posterior={'a': np.random.randn(4, 100)}, posterior_predictive={'y': post_pred}, observed_data={'y': obs}, log_likelihood={'y': -post_pred ** 2, 'decoy': np.zeros_like(post_pred)})
    loo_pit_data = loo_pit(idata, y='y')
    assert np.all((loo_pit_data >= 0) & (loo_pit_data <= 1))