from __future__ import annotations
import builtins
from collections import (
from collections.abc import (
import contextlib
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import iterable_not_string
def index_labels_to_array(labels: np.ndarray | Iterable, dtype: NpDtype | None=None) -> np.ndarray:
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    dtype : dtype
        If specified, use as dtype of the resulting array, otherwise infer.

    Returns
    -------
    array
    """
    if isinstance(labels, (str, tuple)):
        labels = [labels]
    if not isinstance(labels, (list, np.ndarray)):
        try:
            labels = list(labels)
        except TypeError:
            labels = [labels]
    labels = asarray_tuplesafe(labels, dtype=dtype)
    return labels