import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
def inplace_predict(client: Optional['distributed.Client'], model: Union[TrainReturnT, Booster, 'distributed.Future'], data: _DataT, iteration_range: Tuple[int, int]=(0, 0), predict_type: str='value', missing: float=numpy.nan, validate_features: bool=True, base_margin: Optional[_DaskCollection]=None, strict_shape: bool=False) -> Any:
    """Inplace prediction. See doc in :py:meth:`xgboost.Booster.inplace_predict` for
    details.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        See :py:func:`xgboost.dask.predict` for details.
    data :
        dask collection.
    iteration_range:
        See :py:meth:`xgboost.Booster.predict` for details.
    predict_type:
        See :py:meth:`xgboost.Booster.inplace_predict` for details.
    missing:
        Value in the input data which needs to be present as a missing
        value. If None, defaults to np.nan.
    base_margin:
        See :py:obj:`xgboost.DMatrix` for details.

        .. versionadded:: 1.4.0

    strict_shape:
        See :py:meth:`xgboost.Booster.predict` for details.

        .. versionadded:: 1.4.0

    Returns
    -------
    prediction :
        When input data is ``dask.array.Array``, the return value is an array, when
        input data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    """
    _assert_dask_support()
    client = _xgb_get_client(client)
    return client.sync(_inplace_predict_async, global_config=config.get_config(), **locals())