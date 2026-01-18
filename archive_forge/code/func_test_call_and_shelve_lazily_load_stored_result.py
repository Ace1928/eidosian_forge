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
def test_call_and_shelve_lazily_load_stored_result(tmpdir):
    """Check call_and_shelve only load stored data if needed."""
    test_access_time_file = tmpdir.join('test_access')
    test_access_time_file.write('test_access')
    test_access_time = os.stat(test_access_time_file.strpath).st_atime
    time.sleep(0.5)
    assert test_access_time_file.read() == 'test_access'
    if test_access_time == os.stat(test_access_time_file.strpath).st_atime:
        pytest.skip('filesystem does not support fine-grained access time attribute')
    memory = Memory(location=tmpdir.strpath, verbose=0)
    func = memory.cache(f)
    func_id, argument_hash = func._get_output_identifiers(2)
    result_path = os.path.join(memory.store_backend.location, func_id, argument_hash, 'output.pkl')
    assert func(2) == 5
    first_access_time = os.stat(result_path).st_atime
    time.sleep(1)
    result = func.call_and_shelve(2)
    assert isinstance(result, MemorizedResult)
    assert os.stat(result_path).st_atime == first_access_time
    time.sleep(1)
    assert result.get() == 5
    assert os.stat(result_path).st_atime > first_access_time