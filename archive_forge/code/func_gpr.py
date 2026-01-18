from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def gpr(data, xseq, **params):
    """
    Fit gaussian process
    """
    try:
        from sklearn import gaussian_process
    except ImportError as e:
        msg = 'To use gaussian process smoothing, You need to install scikit-learn.'
        raise PlotnineError(msg) from e
    kwargs = params['method_args']
    if not kwargs:
        warnings.warn("See sklearn.gaussian_process.GaussianProcessRegressor for parameters to pass in as 'method_args'", PlotnineWarning)
    regressor = gaussian_process.GaussianProcessRegressor(**kwargs)
    X = np.atleast_2d(data['x']).T
    n = len(data)
    Xseq = np.atleast_2d(xseq).T
    regressor.fit(X, data['y'])
    data = pd.DataFrame({'x': xseq})
    if params['se']:
        y, stderr = regressor.predict(Xseq, return_std=True)
        data['y'] = y
        data['se'] = stderr
        data['ymin'], data['ymax'] = tdist_ci(y, n - 1, stderr, params['level'])
    else:
        data['y'] = regressor.predict(Xseq, return_std=True)
    return data