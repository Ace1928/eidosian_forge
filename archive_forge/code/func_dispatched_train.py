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
def dispatched_train(parameters: Dict, rabit_args: Dict[str, Union[str, int]], train_id: int, evals_name: List[str], evals_id: List[int], train_ref: dict, *refs: dict) -> Optional[TrainReturnT]:
    worker = distributed.get_worker()
    local_param = parameters.copy()
    n_threads = 0
    dwnt = worker.state.nthreads if hasattr(worker, 'state') else worker.nthreads
    for p in ['nthread', 'n_jobs']:
        if local_param.get(p, None) is not None and local_param.get(p, dwnt) != dwnt:
            LOGGER.info('Overriding `nthreads` defined in dask worker.')
            n_threads = local_param[p]
            break
    if n_threads == 0 or n_threads is None:
        n_threads = dwnt
    local_param.update({'nthread': n_threads, 'n_jobs': n_threads})
    local_history: TrainingCallback.EvalsLog = {}
    with CommunicatorContext(**rabit_args), config.config_context(**global_config):
        Xy = _dmatrix_from_list_of_parts(**train_ref, nthread=n_threads)
        evals: List[Tuple[DMatrix, str]] = []
        for i, ref in enumerate(refs):
            if evals_id[i] == train_id:
                evals.append((Xy, evals_name[i]))
                continue
            if ref.get('ref', None) is not None:
                if ref['ref'] != train_id:
                    raise ValueError('The training DMatrix should be used as a reference to evaluation `QuantileDMatrix`.')
                del ref['ref']
                eval_Xy = _dmatrix_from_list_of_parts(**ref, nthread=n_threads, ref=Xy)
            else:
                eval_Xy = _dmatrix_from_list_of_parts(**ref, nthread=n_threads)
            evals.append((eval_Xy, evals_name[i]))
        booster = worker_train(params=local_param, dtrain=Xy, num_boost_round=num_boost_round, evals_result=local_history, evals=evals if len(evals) != 0 else None, obj=obj, feval=feval, custom_metric=custom_metric, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval, xgb_model=xgb_model, callbacks=callbacks)
        return _filter_empty(booster, local_history, Xy.num_row() != 0)