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
def ltr_metric_decorator(func: Callable, n_jobs: Optional[int]) -> Metric:
    """Decorate a learning to rank metric."""

    def inner(y_score: np.ndarray, dmatrix: DMatrix) -> Tuple[str, float]:
        y_true = dmatrix.get_label()
        group_ptr = dmatrix.get_uint_info('group_ptr')
        if group_ptr.size < 2:
            raise ValueError('Invalid `group_ptr`. Likely caused by invalid qid or group.')
        scores = np.empty(group_ptr.size - 1)
        futures = []
        weight = dmatrix.get_group()
        no_weight = weight.size == 0

        def task(i: int) -> float:
            begin = group_ptr[i - 1]
            end = group_ptr[i]
            gy = y_true[begin:end]
            gp = y_score[begin:end]
            if gy.size == 1:
                return 1.0
            return func(gy, gp)
        workers = n_jobs if n_jobs is not None else os.cpu_count()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(1, group_ptr.size):
                f = executor.submit(task, i)
                futures.append(f)
            for i, f in enumerate(futures):
                scores[i] = f.result()
        if no_weight:
            return (func.__name__, scores.mean())
        return (func.__name__, np.average(scores, weights=weight))
    return inner