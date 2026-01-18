from statsmodels.compat.platform import PLATFORM_OSX
from statsmodels.regression.process_regression import (
import numpy as np
import pandas as pd
import pytest
import collections
import statsmodels.tools.numdiff as nd
from numpy.testing import assert_allclose, assert_equal
def run_formula(n, get_model, noise):
    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise)
    df = pd.DataFrame({'y': y, 'x1': x_mean[:, 0], 'x2': x_mean[:, 1], 'x3': x_mean[:, 2], 'x4': x_mean[:, 3], 'xsc1': x_sc[:, 0], 'xsc2': x_sc[:, 1], 'xsm1': x_sm[:, 0], 'xsm2': x_sm[:, 1], 'time': time, 'groups': groups})
    if noise:
        df['xno1'] = x_no[:, 0]
        df['xno2'] = x_no[:, 1]
    mean_formula = 'y ~ 0 + x1 + x2 + x3 + x4'
    scale_formula = '0 + xsc1 + xsc2'
    smooth_formula = '0 + xsm1 + xsm2'
    if noise:
        noise_formula = '0 + xno1 + xno2'
    else:
        noise_formula = None
    preg = ProcessMLE.from_formula(mean_formula, data=df, scale_formula=scale_formula, smooth_formula=smooth_formula, noise_formula=noise_formula, time='time', groups='groups')
    f = preg.fit()
    return (f, df)