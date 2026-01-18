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
def as_arrays(self) -> Tuple[Union[np.ndarray, sparse.csr_matrix], ArrayLike, Optional[ArrayLike]]:
    if isinstance(self.X[0], sparse.csr_matrix):
        X = sparse.vstack(self.X, format='csr')
    else:
        X = np.concatenate(self.X, axis=0)
    y = np.concatenate(self.y, axis=0)
    if self.w:
        w = np.concatenate(self.w, axis=0)
    else:
        w = None
    return (X, y, w)