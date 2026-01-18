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
@pytest.mark.parametrize('incompatibility', ['y-y_hat1', 'y-y_hat2', 'y_hat-log_weights'])
def test_loo_pit_bad_input_shape(incompatibility):
    """Test shape incompatibilities."""
    y = np.random.random(8)
    y_hat = np.random.random((8, 200))
    log_weights = np.random.random((8, 200))
    if incompatibility == 'y-y_hat1':
        with pytest.raises(ValueError, match='1 more dimension'):
            loo_pit(y=y, y_hat=y_hat[None, :], log_weights=log_weights)
    elif incompatibility == 'y-y_hat2':
        with pytest.raises(ValueError, match='y has shape'):
            loo_pit(y=y, y_hat=y_hat[1:3, :], log_weights=log_weights)
    elif incompatibility == 'y_hat-log_weights':
        with pytest.raises(ValueError, match='must have the same shape'):
            loo_pit(y=y, y_hat=y_hat[:, :100], log_weights=log_weights)