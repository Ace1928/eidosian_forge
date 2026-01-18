from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def lm(data, xseq, **params):
    """
    Fit OLS / WLS if data has weight
    """
    import statsmodels.api as sm
    if params['formula']:
        return lm_formula(data, xseq, **params)
    X = sm.add_constant(data['x'])
    Xseq = sm.add_constant(xseq)
    weights = data.get('weights', None)
    if weights is None:
        init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.OLS, sm.OLS.fit)
        model = sm.OLS(data['y'], X, **init_kwargs)
    else:
        if np.any(weights < 0):
            raise ValueError('All weights must be greater than zero.')
        init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.WLS, sm.WLS.fit)
        model = sm.WLS(data['y'], X, weights=data['weight'], **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(Xseq)
    if params['se']:
        alpha = 1 - params['level']
        prstd, iv_l, iv_u = wls_prediction_std(results, Xseq, alpha=alpha)
        data['se'] = prstd
        data['ymin'] = iv_l
        data['ymax'] = iv_u
    return data