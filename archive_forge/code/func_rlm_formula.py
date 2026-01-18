from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def rlm_formula(data, xseq, **params):
    """
    Fit RLM using a formula
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    eval_env = _to_patsy_env(params['enviroment'])
    formula = params['formula']
    init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.RLM, sm.RLM.fit)
    model = smf.rlm(formula, data, eval_env=eval_env, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(data)
    if params['se']:
        warnings.warn('Confidence intervals are not yet implemented for RLM smoothing.', PlotnineWarning)
    return data