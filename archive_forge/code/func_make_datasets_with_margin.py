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
def make_datasets_with_margin(unweighted_strategy: strategies.SearchStrategy) -> Callable:
    """Factory function for creating strategies that generates datasets with weight and
    base margin.

    """

    @strategies.composite
    def weight_margin(draw: Callable) -> TestDataset:
        data: TestDataset = draw(unweighted_strategy)
        if draw(strategies.booleans()):
            data.w = draw(arrays(np.float64, len(data.y), elements=strategies.floats(0.1, 2.0)))
        if draw(strategies.booleans()):
            num_class = 1
            if data.objective == 'multi:softmax':
                num_class = int(np.max(data.y) + 1)
            elif data.name.startswith('mtreg'):
                num_class = data.y.shape[1]
            data.margin = draw(arrays(np.float64, data.y.shape[0] * num_class, elements=strategies.floats(0.5, 1.0)))
            assert data.margin is not None
            if num_class != 1:
                data.margin = data.margin.reshape(data.y.shape[0], num_class)
        return data
    return weight_margin