import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
@pytest.mark.smoke
def test_getframe_smoke():
    mod = Factor(X.iloc[:, 1:-1], 2, smc=True)
    res = mod.fit()
    df = res.get_loadings_frame(style='raw')
    assert_(isinstance(df, pd.DataFrame))
    lds = res.get_loadings_frame(style='strings', decimals=3, threshold=0.3)
    try:
        from jinja2 import Template
    except ImportError:
        return
    if PD_LT_1_4:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            lds.to_latex()
    else:
        lds.style.to_latex()
    try:
        from pandas.io import formats as pd_formats
    except ImportError:
        from pandas import formats as pd_formats
    ldf = res.get_loadings_frame(style='display')
    assert_(isinstance(ldf, pd_formats.style.Styler))
    assert_(isinstance(ldf.data, pd.DataFrame))
    res.get_loadings_frame(style='display', decimals=3, threshold=0.2)
    res.get_loadings_frame(style='display', decimals=3, color_max='GAINSBORO')
    res.get_loadings_frame(style='display', decimals=3, threshold=0.45, highlight_max=False, sort_=False)