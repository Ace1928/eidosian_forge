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
def test_memory_objects_repr(tmpdir):

    def my_func(a, b):
        return a + b
    memory = Memory(location=tmpdir.strpath, verbose=0)
    memorized_func = memory.cache(my_func)
    memorized_func_repr = 'MemorizedFunc(func={func}, location={location})'
    assert str(memorized_func) == memorized_func_repr.format(func=my_func, location=memory.store_backend.location)
    memorized_result = memorized_func.call_and_shelve(42, 42)
    memorized_result_repr = 'MemorizedResult(location="{location}", func="{func}", args_id="{args_id}")'
    assert str(memorized_result) == memorized_result_repr.format(location=memory.store_backend.location, func=memorized_result.func_id, args_id=memorized_result.args_id)
    assert str(memory) == 'Memory(location={location})'.format(location=memory.store_backend.location)