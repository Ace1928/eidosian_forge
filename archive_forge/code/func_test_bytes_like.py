from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
def test_bytes_like():
    assert 1000 <= sizeof(bytes(1000)) <= 2000
    assert 1000 <= sizeof(bytearray(1000)) <= 2000
    assert 1000 <= sizeof(memoryview(bytes(1000))) <= 2000
    assert 8000 <= sizeof(array('d', range(1000))) <= 9000