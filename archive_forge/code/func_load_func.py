import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import (
def load_func(expected, filename):
    for i in range(10):
        try:
            with open(filename, 'rb') as f:
                reloaded = cpickle.load(f)
            break
        except (OSError, IOError):
            time.sleep(0.1)
    else:
        raise
    assert expected == reloaded