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
def eval_error_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, np.float64]:
    """Evaluation metric for xgb.train"""
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    if predt.size == 0:
        return ('CustomErr', np.float64(0.0))
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return ('CustomErr', np.sum(r))