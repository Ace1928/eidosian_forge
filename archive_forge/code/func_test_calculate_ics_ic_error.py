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
def test_calculate_ics_ic_error(centered_eight, non_centered_eight):
    in_dict = {'centered': loo(centered_eight), 'non_centered': waic(non_centered_eight)}
    with pytest.raises(ValueError, match='found both loo and waic'):
        _calculate_ics(in_dict)