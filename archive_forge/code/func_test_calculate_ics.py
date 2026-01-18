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
def test_calculate_ics(centered_eight, non_centered_eight, ic):
    ic_func = loo if ic == 'loo' else waic
    idata_dict = {'centered': centered_eight, 'non_centered': non_centered_eight}
    elpddata_dict = {key: ic_func(value) for key, value in idata_dict.items()}
    mixed_dict = {'centered': idata_dict['centered'], 'non_centered': elpddata_dict['non_centered']}
    idata_out, _, _ = _calculate_ics(idata_dict, ic=ic)
    elpddata_out, _, _ = _calculate_ics(elpddata_dict, ic=ic)
    mixed_out, _, _ = _calculate_ics(mixed_dict, ic=ic)
    for model in idata_dict:
        ic_ = f'elpd_{ic}'
        assert idata_out[model][ic_] == elpddata_out[model][ic_]
        assert idata_out[model][ic_] == mixed_out[model][ic_]
        assert idata_out[model][f'p_{ic}'] == elpddata_out[model][f'p_{ic}']
        assert idata_out[model][f'p_{ic}'] == mixed_out[model][f'p_{ic}']