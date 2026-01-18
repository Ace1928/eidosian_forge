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
def test_file_manager_cache_and_refcounts() -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    cache: dict = {}
    ref_counts: dict = {}
    manager = CachingFileManager(opener, 'filename', cache=cache, ref_counts=ref_counts)
    assert ref_counts[manager._key] == 1
    assert not cache
    manager.acquire()
    assert len(cache) == 1
    with set_options(warn_for_unclosed_files=False):
        del manager
        gc.collect()
    assert not ref_counts
    assert not cache