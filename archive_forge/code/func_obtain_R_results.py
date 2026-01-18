from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def obtain_R_results(path):
    with path.open('r', encoding='utf-8') as f:
        R_results = json.load(f)
    results = {}
    for damped in R_results:
        new_key = damped == 'TRUE'
        results[new_key] = {}
        for model in R_results[damped]:
            if len(R_results[damped][model]):
                results[new_key][model] = R_results[damped][model]
    for damped in results:
        for model in results[damped]:
            for key in ['alpha', 'beta', 'gamma', 'phi', 'sigma2']:
                results[damped][model][key] = float(results[damped][model][key][0])
            for key in ['states', 'initstate', 'residuals', 'fitted', 'forecast', 'simulation']:
                results[damped][model][key] = np.asarray(results[damped][model][key])
    return results