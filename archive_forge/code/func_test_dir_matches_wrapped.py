from __future__ import annotations
import importlib
import io
import os
import re
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import sentinel
import pytest
import trio
from trio import _core, _file_io
from trio._file_io import _FILE_ASYNC_METHODS, _FILE_SYNC_ATTRS, AsyncIOWrapper
def test_dir_matches_wrapped(async_file: AsyncIOWrapper[mock.Mock], wrapped: mock.Mock) -> None:
    attrs = _FILE_SYNC_ATTRS.union(_FILE_ASYNC_METHODS)
    assert all((attr in dir(async_file) for attr in attrs if attr in dir(wrapped)))
    assert not any((attr in dir(async_file) for attr in attrs if attr not in dir(wrapped)))