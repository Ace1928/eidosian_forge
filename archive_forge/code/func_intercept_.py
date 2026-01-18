import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
@property
def intercept_(self) -> np.ndarray:
    """
        Intercept (bias) property

        .. note:: Intercept is defined only for linear learners

            Intercept (bias) is only defined when the linear model is chosen as base
            learner (`booster=gblinear`). It is not defined for other base learner types,
            such as tree learners (`booster=gbtree`).

        Returns
        -------
        intercept_ : array of shape ``(1,)`` or ``[n_classes]``
        """
    if self.get_xgb_params()['booster'] != 'gblinear':
        raise AttributeError(f'Intercept (bias) is not defined for Booster type {self.booster}')
    b = self.get_booster()
    return np.array(json.loads(b.get_dump(dump_format='json')[0])['bias'])