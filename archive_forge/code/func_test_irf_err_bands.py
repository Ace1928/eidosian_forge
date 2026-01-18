from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
@pytest.mark.slow
def test_irf_err_bands():
    data = get_macrodata()
    model = VAR(data)
    results = model.fit(maxlags=2)
    irf = results.irf()
    bands_sz1 = irf.err_band_sz1()
    bands_sz2 = irf.err_band_sz2()
    bands_sz3 = irf.err_band_sz3()
    bands_mc = irf.errband_mc()