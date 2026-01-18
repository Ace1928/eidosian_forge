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
def test_file_manager_write_consecutive(tmpdir, file_cache) -> None:
    path1 = str(tmpdir.join('testing1.txt'))
    path2 = str(tmpdir.join('testing2.txt'))
    manager1 = CachingFileManager(open, path1, mode='w', cache=file_cache)
    manager2 = CachingFileManager(open, path2, mode='w', cache=file_cache)
    f1a = manager1.acquire()
    f1a.write('foo')
    f1a.flush()
    f2 = manager2.acquire()
    f2.write('bar')
    f2.flush()
    f1b = manager1.acquire()
    f1b.write('baz')
    assert (getattr(file_cache, 'maxsize', float('inf')) > 1) == (f1a is f1b)
    manager1.close()
    manager2.close()
    with open(path1) as f:
        assert f.read() == 'foobaz'
    with open(path2) as f:
        assert f.read() == 'bar'