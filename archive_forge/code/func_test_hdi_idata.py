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
def test_hdi_idata(centered_eight):
    data = centered_eight.posterior
    result = hdi(data)
    assert isinstance(result, Dataset)
    assert dict(result.sizes) == {'school': 8, 'hdi': 2}
    result = hdi(data, input_core_dims=[['chain']])
    assert isinstance(result, Dataset)
    assert result.sizes == {'draw': 500, 'hdi': 2, 'school': 8}