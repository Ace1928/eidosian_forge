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
@pytest.mark.parametrize('memory_kwargs', [{'compress': 3, 'verbose': 2}, {'mmap_mode': 'r', 'verbose': 5, 'backend_options': {'parameter': 'unused'}}])
def test_memory_pickle_dump_load(tmpdir, memory_kwargs):
    memory = Memory(location=tmpdir.strpath, **memory_kwargs)
    memory_reloaded = pickle.loads(pickle.dumps(memory))
    compare(memory.store_backend, memory_reloaded.store_backend)
    compare(memory, memory_reloaded, ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(memory) == hash(memory_reloaded)
    func_cached = memory.cache(f)
    func_cached_reloaded = pickle.loads(pickle.dumps(func_cached))
    compare(func_cached.store_backend, func_cached_reloaded.store_backend)
    compare(func_cached, func_cached_reloaded, ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(func_cached) == hash(func_cached_reloaded)
    memorized_result = func_cached.call_and_shelve(1)
    memorized_result_reloaded = pickle.loads(pickle.dumps(memorized_result))
    compare(memorized_result.store_backend, memorized_result_reloaded.store_backend)
    compare(memorized_result, memorized_result_reloaded, ignored_attrs=set(['store_backend', 'timestamp', '_func_code_id']))
    assert hash(memorized_result) == hash(memorized_result_reloaded)