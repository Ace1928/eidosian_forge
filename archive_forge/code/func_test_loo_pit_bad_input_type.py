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
@pytest.mark.parametrize('arg', ['y', 'y_hat', 'log_weights'])
def test_loo_pit_bad_input_type(centered_eight, arg):
    """Test wrong input type (not None, str not DataArray."""
    kwargs = {'y': 'obs', 'y_hat': 'obs', 'log_weights': None}
    kwargs[arg] = 2
    with pytest.raises(ValueError, match=f'not {type(2)}'):
        loo_pit(idata=centered_eight, **kwargs)