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
@pytest.mark.parametrize('ic', ['loo', 'waic'])
@pytest.mark.parametrize('method', ['stacking', 'BB-pseudo-BMA', 'pseudo-BMA'])
def test_compare_different_multidim(multidim_models, ic, method):
    model_dict = {'model_1': multidim_models.model_1, 'model_2': multidim_models.model_2}
    weight = compare(model_dict, ic=ic, method=method)['weight']
    assert weight['model_1'] > weight['model_2']
    assert_allclose(np.sum(weight), 1.0)