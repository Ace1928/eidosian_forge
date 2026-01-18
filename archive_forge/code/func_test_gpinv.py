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
@pytest.mark.parametrize('probs', [True, False])
@pytest.mark.parametrize('kappa', [-1, -0.5, 1e-30, 0.5, 1])
@pytest.mark.parametrize('sigma', [0, 2])
def test_gpinv(probs, kappa, sigma):
    if probs:
        probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3])
    else:
        probs = np.array([-0.1, 0.1, 0.1, 0.2, 0.3])
    assert len(_gpinv(probs, kappa, sigma)) == len(probs)