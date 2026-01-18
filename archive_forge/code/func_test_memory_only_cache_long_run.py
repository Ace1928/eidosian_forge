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
def test_memory_only_cache_long_run(self, memory):
    """Test cache validity based on run duration."""

    def cache_validation_callback(metadata):
        duration = metadata['duration']
        if duration > 0.1:
            return True
    f = memory.cache(self.foo, cache_validation_callback=cache_validation_callback, ignore=['d'])
    d1, d2 = ({'run': False}, {'run': False})
    assert f(2, d1, delay=0) == 4
    assert f(2, d2, delay=0) == 4
    assert d1['run']
    assert d2['run']
    d1, d2 = ({'run': False}, {'run': False})
    assert f(2, d1, delay=0.2) == 4
    assert f(2, d2, delay=0.2) == 4
    assert d1['run']
    assert not d2['run']