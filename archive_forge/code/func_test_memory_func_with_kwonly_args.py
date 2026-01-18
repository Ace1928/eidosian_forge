import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash
def test_memory_func_with_kwonly_args(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    func_cached = memory.cache(func_with_kwonly_args)
    assert func_cached(1, 2, kw1=3) == (1, 2, 3, 'kw2')
    with raises(ValueError) as excinfo:
        func_cached(1, 2, 3, kw2=4)
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional parameter")
    func_cached(1, 2, kw1=3, kw2=4)
    with raises(ValueError) as excinfo:
        func_cached(1, 2, 3, kw2=4)
    excinfo.match("Keyword-only parameter 'kw1' was passed as positional parameter")
    func_cached = memory.cache(func_with_kwonly_args, ignore=['kw2'])
    assert func_cached(1, 2, kw1=3, kw2=4) == (1, 2, 3, 4)
    assert func_cached(1, 2, kw1=3, kw2='ignored') == (1, 2, 3, 4)