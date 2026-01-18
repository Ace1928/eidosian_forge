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
def test_hdi_idata_varnames(centered_eight):
    data = centered_eight.posterior
    result = hdi(data, var_names=['mu', 'theta'])
    assert isinstance(result, Dataset)
    assert result.sizes == {'hdi': 2, 'school': 8}
    assert list(result.data_vars.keys()) == ['mu', 'theta']