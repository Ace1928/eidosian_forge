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
def test_apply_test_function_missing_group():
    """Test error when InferenceData object is missing a required group.

    The function cannot work if group="both" but InferenceData object has no
    posterior_predictive group.
    """
    idata = from_dict(posterior={'a': np.random.random((4, 500, 30))}, observed_data={'y': np.random.random(30)})
    with pytest.raises(ValueError, match='must have posterior_predictive'):
        apply_test_function(idata, lambda y, theta: np.mean, group='both')