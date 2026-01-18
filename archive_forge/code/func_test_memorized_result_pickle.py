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
def test_memorized_result_pickle(tmpdir):
    memory = Memory(location=tmpdir.strpath)

    @memory.cache
    def g(x):
        return x ** 2
    memorized_result = g.call_and_shelve(4)
    memorized_result_pickle = pickle.dumps(memorized_result)
    memorized_result_loads = pickle.loads(memorized_result_pickle)
    assert memorized_result.store_backend.location == memorized_result_loads.store_backend.location
    assert memorized_result.func == memorized_result_loads.func
    assert memorized_result.args_id == memorized_result_loads.args_id
    assert str(memorized_result) == str(memorized_result_loads)