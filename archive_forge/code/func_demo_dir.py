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
def demo_dir(path: str) -> str:
    """Look for the demo directory based on the test file name."""
    path = normpath(os.path.dirname(path))
    while True:
        subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
        subdirs = [os.path.basename(d) for d in subdirs]
        if 'demo' in subdirs:
            return os.path.join(path, 'demo')
        new_path = normpath(os.path.join(path, os.path.pardir))
        assert new_path != path
        path = new_path