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
def test_hdi_skipna():
    normal_sample = np.random.randn(500)
    interval = hdi(normal_sample[10:])
    normal_sample[:10] = np.nan
    interval_ = hdi(normal_sample, skipna=True)
    assert_array_almost_equal(interval, interval_)