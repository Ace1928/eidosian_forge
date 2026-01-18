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
@pytest.mark.parametrize('pointwise', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('kwargs', [{}, {'group': 'posterior_predictive', 'var_names': {'posterior_predictive': 'obs'}}, {'group': 'observed_data', 'var_names': {'both': 'obs'}, 'out_data_shape': 'shape'}, {'var_names': {'both': 'obs', 'posterior': ['theta', 'mu']}}, {'group': 'observed_data', 'out_name_data': 'T_name'}])
def test_apply_test_function(centered_eight, pointwise, inplace, kwargs):
    """Test some usual call cases of apply_test_function"""
    centered_eight = deepcopy(centered_eight)
    group = kwargs.get('group', 'both')
    var_names = kwargs.get('var_names', None)
    out_data_shape = kwargs.get('out_data_shape', None)
    out_pp_shape = kwargs.get('out_pp_shape', None)
    out_name_data = kwargs.get('out_name_data', 'T')
    if out_data_shape == 'shape':
        out_data_shape = (8,) if pointwise else ()
    if out_pp_shape == 'shape':
        out_pp_shape = (4, 500, 8) if pointwise else (4, 500)
    idata = deepcopy(centered_eight)
    idata_out = apply_test_function(idata, lambda y, theta: np.mean(y), group=group, var_names=var_names, pointwise=pointwise, out_name_data=out_name_data, out_data_shape=out_data_shape, out_pp_shape=out_pp_shape)
    if inplace:
        assert idata is idata_out
    if group == 'both':
        test_dict = {'observed_data': ['T'], 'posterior_predictive': ['T']}
    else:
        test_dict = {group: [kwargs.get('out_name_data', 'T')]}
    fails = check_multiple_attrs(test_dict, idata_out)
    assert not fails