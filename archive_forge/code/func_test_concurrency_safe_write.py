import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import (
@timeout(0)
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky', 'threading'])
def test_concurrency_safe_write(tmpdir, backend):
    filename = tmpdir.join('test.pkl').strpath
    obj = {str(i): i for i in range(int(100000.0))}
    funcs = [functools.partial(concurrency_safe_write_rename, write_func=write_func) if i % 3 != 2 else load_func for i in range(12)]
    Parallel(n_jobs=2, backend=backend)((delayed(func)(obj, filename) for func in funcs))