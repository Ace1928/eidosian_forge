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
def evals_result(self) -> Dict[str, Dict[str, List[float]]]:
    """Return the evaluation results.

        If **eval_set** is passed to the :py:meth:`fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.  When
        **eval_metric** is also passed to the :py:meth:`fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the :py:meth:`fit`
        function.

        The returned evaluation result is a dictionary:

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
             'validation_1': {'logloss': ['0.41965', '0.17686']}}

        Returns
        -------
        evals_result

        """
    if getattr(self, 'evals_result_', None) is not None:
        evals_result = self.evals_result_
    else:
        raise XGBoostError('No evaluation result, `eval_set` is not used during training.')
    return evals_result