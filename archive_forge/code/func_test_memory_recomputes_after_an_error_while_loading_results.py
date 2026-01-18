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
def test_memory_recomputes_after_an_error_while_loading_results(tmpdir, monkeypatch):
    memory = Memory(location=tmpdir.strpath)

    def func(arg):
        time.sleep(0.01)
        return (arg, time.time())
    cached_func = memory.cache(func)
    input_arg = 'arg'
    arg, timestamp = cached_func(input_arg)
    assert arg == input_arg
    corrupt_single_cache_item(memory)
    recorded_warnings = monkeypatch_cached_func_warn(cached_func, monkeypatch)
    recomputed_arg, recomputed_timestamp = cached_func(arg)
    assert len(recorded_warnings) == 1
    exception_msg = 'Exception while loading results'
    assert exception_msg in recorded_warnings[0]
    assert recomputed_arg == arg
    assert recomputed_timestamp > timestamp
    corrupt_single_cache_item(memory)
    reference = cached_func.call_and_shelve(arg)
    try:
        reference.get()
        raise AssertionError('It normally not possible to load a corrupted MemorizedResult')
    except KeyError as e:
        message = 'is corrupted'
        assert message in str(e.args)