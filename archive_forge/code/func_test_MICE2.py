import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_MICE2(self):
    from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
    df = gendat()
    imp_data = mice.MICEData(df)
    mi = mice.MICE('x3 ~ x1 + x2', sm.GLM, imp_data, init_kwds={'family': sm.families.Binomial()})
    for j in range(3):
        x = mi.next_sample()
        assert isinstance(x, GLMResultsWrapper)
        assert isinstance(x.family, sm.families.Binomial)