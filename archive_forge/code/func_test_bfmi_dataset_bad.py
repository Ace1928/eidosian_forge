import os
import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal
from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
def test_bfmi_dataset_bad(self):
    data = load_arviz_data('centered_eight')
    del data.sample_stats['energy']
    with pytest.raises(TypeError):
        bfmi(data)