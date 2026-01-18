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
def test_hdp_2darray():
    normal_sample = np.random.randn(12000, 5)
    msg = 'hdi currently interprets 2d data as \\(draw, shape\\) but this will change in a future release to \\(chain, draw\\) for coherence with other functions'
    with pytest.warns(FutureWarning, match=msg):
        result = hdi(normal_sample)
    assert result.shape == (5, 2)