from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def predictdf(data, xseq, **params):
    """
    Make prediction on the data

    This is a general function responsible for dispatching
    to functions that do predictions for the specific models.
    """
    methods = {'lm': lm, 'ols': lm, 'wls': lm, 'rlm': rlm, 'glm': glm, 'gls': gls, 'lowess': lowess, 'loess': loess, 'mavg': mavg, 'gpr': gpr}
    method = params['method']
    if isinstance(method, str):
        try:
            method = methods[method]
        except KeyError as e:
            msg = f'Method should be one of {list(methods.keys())}'
            raise PlotnineError(msg) from e
    if not callable(method):
        msg = "'method' should either be a string or a functionwith the signature `func(data, xseq, **params)`"
        raise PlotnineError(msg)
    return method(data, xseq, **params)