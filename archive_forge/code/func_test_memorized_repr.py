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
def test_memorized_repr(tmpdir):
    func = MemorizedFunc(f, tmpdir.strpath)
    result = func.call_and_shelve(2)
    func2 = MemorizedFunc(f, tmpdir.strpath)
    result2 = func2.call_and_shelve(2)
    assert result.get() == result2.get()
    assert repr(func) == repr(func2)
    func = NotMemorizedFunc(f)
    repr(func)
    repr(func.call_and_shelve(2))
    func = MemorizedFunc(f, tmpdir.strpath, verbose=11, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()
    func = MemorizedFunc(f, tmpdir.strpath, verbose=11)
    result = func.call_and_shelve(11)
    result.get()
    func = MemorizedFunc(f, tmpdir.strpath, verbose=5, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()
    func = MemorizedFunc(f, tmpdir.strpath, verbose=5)
    result = func.call_and_shelve(11)
    result.get()