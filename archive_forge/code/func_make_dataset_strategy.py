import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
@memory.cache
def make_dataset_strategy() -> Callable:
    _unweighted_datasets_strategy = strategies.sampled_from([TestDataset('calif_housing', get_california_housing, 'reg:squarederror', 'rmse'), TestDataset('calif_housing-l1', get_california_housing, 'reg:absoluteerror', 'mae'), TestDataset('cancer', get_cancer, 'binary:logistic', 'logloss'), TestDataset('sparse', get_sparse, 'reg:squarederror', 'rmse'), TestDataset('sparse-l1', get_sparse, 'reg:absoluteerror', 'mae'), TestDataset('empty', lambda: (np.empty((0, 100)), np.empty(0)), 'reg:squarederror', 'rmse')])
    return make_datasets_with_margin(_unweighted_datasets_strategy)()