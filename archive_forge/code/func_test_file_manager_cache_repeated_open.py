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
def test_file_manager_cache_repeated_open() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    cache: dict = {}
    manager = CachingFileManager(opener, 'filename', cache=cache)
    manager.acquire()
    assert len(cache) == 1
    manager2 = CachingFileManager(opener, 'filename', cache=cache)
    manager2.acquire()
    assert len(cache) == 2
    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()
    assert len(cache) == 1
    with set_options(warn_for_unclosed_files=False):
        del manager2
        gc.collect()
    assert not cache