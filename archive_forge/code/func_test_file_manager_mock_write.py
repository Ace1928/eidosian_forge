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
def test_file_manager_mock_write(file_cache) -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)
    lock = mock.MagicMock(spec=threading.Lock())
    manager = CachingFileManager(opener, 'filename', lock=lock, cache=file_cache)
    f = manager.acquire()
    f.write('contents')
    manager.close()
    assert not file_cache
    opener.assert_called_once_with('filename')
    mock_file.write.assert_called_once_with('contents')
    mock_file.close.assert_called_once_with()
    lock.__enter__.assert_has_calls([mock.call(), mock.call()])