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
def raccoon_face_or_skip():
    if scipy_datasets_require_network:
        run_network_tests = environ.get('SKLEARN_SKIP_NETWORK_TESTS', '1') == '0'
        if not run_network_tests:
            raise SkipTest('test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0')
        try:
            import pooch
        except ImportError:
            raise SkipTest('test requires pooch to be installed')
        from scipy.datasets import face
    else:
        from scipy.misc import face
    return face(gray=True)