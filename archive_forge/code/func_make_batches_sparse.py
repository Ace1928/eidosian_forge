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
def make_batches_sparse(n_samples_per_batch: int, n_features: int, n_batches: int, sparsity: float) -> Tuple[List[sparse.csr_matrix], List[np.ndarray], List[np.ndarray]]:
    X = []
    y = []
    w = []
    rng = np.random.RandomState(1994)
    for _ in range(n_batches):
        _X = sparse.random(n_samples_per_batch, n_features, 1.0 - sparsity, format='csr', dtype=np.float32, random_state=rng)
        _y = rng.randn(n_samples_per_batch)
        _w = rng.uniform(low=0, high=1, size=n_samples_per_batch)
        X.append(_X)
        y.append(_y)
        w.append(_w)
    return (X, y, w)