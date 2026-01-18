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
def test_file_manager_repr() -> None:
    opener = mock.Mock()
    manager = CachingFileManager(opener, 'my-file')
    assert 'my-file' in repr(manager)