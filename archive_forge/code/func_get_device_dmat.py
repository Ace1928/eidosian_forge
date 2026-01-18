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
def get_device_dmat(self, max_bin: Optional[int]) -> xgb.QuantileDMatrix:
    import cupy as cp
    w = None if self.w is None else cp.array(self.w)
    X = cp.array(self.X, dtype=np.float32)
    y = cp.array(self.y, dtype=np.float32)
    return xgb.QuantileDMatrix(X, y, weight=w, base_margin=self.margin, max_bin=max_bin)