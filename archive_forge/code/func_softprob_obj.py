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
def softprob_obj(classes: int) -> SklObjective:

    def objective(labels: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = labels.shape[0]
        grad = np.zeros((rows, classes), dtype=float)
        hess = np.zeros((rows, classes), dtype=float)
        eps = 1e-06
        for r in range(predt.shape[0]):
            target = labels[r]
            p = softmax(predt[r, :])
            for c in range(predt.shape[1]):
                assert target >= 0 or target <= classes
                g = p[c] - 1.0 if c == target else p[c]
                h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
                grad[r, c] = g
                hess[r, c] = h
        grad = grad.reshape((rows * classes, 1))
        hess = hess.reshape((rows * classes, 1))
        return (grad, hess)
    return objective