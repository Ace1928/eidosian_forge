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
def test_memory_in_memory_function_code_change(tmpdir):
    _function_to_cache.__code__ = _sum.__code__
    memory = Memory(location=tmpdir.strpath, verbose=0)
    f = memory.cache(_function_to_cache)
    assert f(1, 2) == 3
    assert f(1, 2) == 3
    with warns(JobLibCollisionWarning):
        _function_to_cache.__code__ = _product.__code__
        assert f(1, 2) == 2
        assert f(1, 2) == 2