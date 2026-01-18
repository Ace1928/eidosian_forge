import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import (
def concurrency_safe_write_rename(to_write, filename, write_func):
    temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    concurrency_safe_rename(temporary_filename, filename)