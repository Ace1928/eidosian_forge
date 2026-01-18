from __future__ import annotations
import gc
import pickle
import threading
from unittest import mock
import pytest
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.lru_cache import LRUCache
from xarray.core.options import set_options
from xarray.tests import assert_no_warnings
def test_file_manager_autoclose_while_locked() -> None:
    opener = mock.Mock()
    lock = threading.Lock()
    cache: dict = {}
    manager = CachingFileManager(opener, 'filename', lock=lock, cache=cache)
    manager.acquire()
    assert cache
    lock.acquire()
    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()
    assert cache