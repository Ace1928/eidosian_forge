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
def test__get_items_to_delete(tmpdir):
    memory, expected_hash_cachedirs, _ = _setup_toy_cache(tmpdir)
    items = memory.store_backend.get_items()
    items_to_delete = memory.store_backend._get_items_to_delete('2K')
    nb_hashes = len(expected_hash_cachedirs)
    assert set.issubset(set(items_to_delete), set(items))
    assert len(items_to_delete) == nb_hashes - 1
    items_to_delete_2048b = memory.store_backend._get_items_to_delete(2048)
    assert sorted(items_to_delete) == sorted(items_to_delete_2048b)
    items_to_delete_empty = memory.store_backend._get_items_to_delete('1M')
    assert items_to_delete_empty == []
    bytes_limit_too_small = 500
    items_to_delete_500b = memory.store_backend._get_items_to_delete(bytes_limit_too_small)
    assert set(items_to_delete_500b), set(items)
    items_to_delete_6000b = memory.store_backend._get_items_to_delete(6000)
    surviving_items = set(items).difference(items_to_delete_6000b)
    assert max((ci.last_access for ci in items_to_delete_6000b)) <= min((ci.last_access for ci in surviving_items))