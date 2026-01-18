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
def mapped_predict(booster: Booster, partition: Any, is_df: bool, columns: List[int], base_margin: Any) -> Any:
    with config.config_context(**global_config):
        prediction = booster.inplace_predict(partition, iteration_range=iteration_range, predict_type=predict_type, missing=missing, base_margin=base_margin, validate_features=validate_features, strict_shape=strict_shape)
    prediction = _maybe_dataframe(partition, prediction, columns, is_df)
    return prediction