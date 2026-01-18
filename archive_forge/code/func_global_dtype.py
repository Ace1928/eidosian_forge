import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest
import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from threadpoolctl import threadpool_limits
from sklearn import config_context, set_config
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.datasets import (
from sklearn.tests import random_seed
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import get_pytest_filterwarning_lines
from sklearn.utils.fixes import (
@pytest.fixture(params=[pytest.param(np.float32, marks=_SKIP32_MARK), np.float64])
def global_dtype(request):
    yield request.param