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
def test_hdi_multimodal():
    normal_sample = np.concatenate((np.random.normal(-4, 1, 2500000), np.random.normal(2, 0.5, 2500000)))
    intervals = hdi(normal_sample, multimodal=True)
    assert_array_almost_equal(intervals, [[-5.8, -2.2], [0.9, 3.1]], 1)