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
@pytest.mark.parametrize('warn_for_unclosed_files', [True, False])
def test_file_manager_autoclose(warn_for_unclosed_files) -> None:
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)
    cache: dict = {}
    manager = CachingFileManager(opener, 'filename', cache=cache)
    manager.acquire()
    assert cache
    if warn_for_unclosed_files:
        ctx = pytest.warns(RuntimeWarning)
    else:
        ctx = assert_no_warnings()
    with set_options(warn_for_unclosed_files=warn_for_unclosed_files):
        with ctx:
            del manager
            gc.collect()
    assert not cache
    mock_file.close.assert_called_once_with()