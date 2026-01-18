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
def test_loo_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood['obs'][:, :, 1] = 10
    with pytest.warns(UserWarning) as records:
        assert loo(centered_eight, pointwise=True) is not None
    assert any(('Estimated shape parameter' in str(record.message) for record in records))
    centered_eight.log_likelihood['obs'][:, :, :] = 1
    with pytest.warns(UserWarning) as records:
        assert loo(centered_eight, pointwise=True) is not None
    assert any(('Estimated shape parameter' in str(record.message) for record in records))