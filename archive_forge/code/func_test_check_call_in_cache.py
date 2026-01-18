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
def test_check_call_in_cache(tmpdir):
    for func in (MemorizedFunc(f, tmpdir.strpath), Memory(location=tmpdir.strpath, verbose=0).cache(f)):
        result = func.check_call_in_cache(2)
        assert not result
        assert isinstance(result, bool)
        assert func(2) == 5
        result = func.check_call_in_cache(2)
        assert result
        assert isinstance(result, bool)
        func.clear()