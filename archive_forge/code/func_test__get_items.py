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
def test__get_items(tmpdir):
    memory, expected_hash_dirs, _ = _setup_toy_cache(tmpdir)
    items = memory.store_backend.get_items()
    hash_dirs = [ci.path for ci in items]
    assert set(hash_dirs) == set(expected_hash_dirs)

    def get_files_size(directory):
        full_paths = [os.path.join(directory, fn) for fn in os.listdir(directory)]
        return sum((os.path.getsize(fp) for fp in full_paths))
    expected_hash_cache_sizes = [get_files_size(hash_dir) for hash_dir in hash_dirs]
    hash_cache_sizes = [ci.size for ci in items]
    assert hash_cache_sizes == expected_hash_cache_sizes
    output_filenames = [os.path.join(hash_dir, 'output.pkl') for hash_dir in hash_dirs]
    expected_last_accesses = [datetime.datetime.fromtimestamp(os.path.getatime(fn)) for fn in output_filenames]
    last_accesses = [ci.last_access for ci in items]
    assert last_accesses == expected_last_accesses