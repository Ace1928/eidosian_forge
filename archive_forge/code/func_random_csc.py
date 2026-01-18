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
def random_csc(t_id: int) -> sparse.csc_matrix:
    rng = np.random.default_rng(1994 * t_id)
    thread_size = n_features // n_threads
    if t_id == n_threads - 1:
        n_features_tloc = n_features - t_id * thread_size
    else:
        n_features_tloc = thread_size
    X = sparse.random(m=n_samples, n=n_features_tloc, density=1.0 - sparsity, random_state=rng).tocsc()
    y = np.zeros((n_samples, 1))
    for i in range(X.shape[1]):
        size = X.indptr[i + 1] - X.indptr[i]
        if size != 0:
            y += X[:, i].toarray() * rng.random((n_samples, 1)) * 0.2
    return (X, y)