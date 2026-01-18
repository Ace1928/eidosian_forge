from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def glm_formula(data, xseq, **params):
    """
    Fit with GLM formula
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    eval_env = _to_patsy_env(params['enviroment'])
    init_kwargs, fit_kwargs = separate_method_kwargs(params['method_args'], sm.GLM, sm.GLM.fit)
    model = smf.glm(params['formula'], data, eval_env=eval_env, **init_kwargs)
    results = model.fit(**fit_kwargs)
    data = pd.DataFrame({'x': xseq})
    data['y'] = results.predict(data)
    if params['se']:
        xdata = pd.DataFrame({'x': xseq})
        prediction = results.get_prediction(xdata)
        ci = prediction.conf_int(1 - params['level'])
        data['ymin'] = ci[:, 0]
        data['ymax'] = ci[:, 1]
    return data