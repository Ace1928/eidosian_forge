import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.fixture
def open_logging_fs(monkeypatch):
    from pyarrow.fs import LocalFileSystem, PyFileSystem
    from .test_fs import ProxyHandler
    localfs = LocalFileSystem()

    def normalized(paths):
        return {localfs.normalize_path(str(p)) for p in paths}
    opened = set()

    def open_input_file(self, path):
        path = localfs.normalize_path(str(path))
        opened.add(path)
        return self._fs.open_input_file(path)
    monkeypatch.setattr(ProxyHandler, 'open_input_file', open_input_file)
    fs = PyFileSystem(ProxyHandler(localfs))

    @contextlib.contextmanager
    def assert_opens(expected_opened):
        opened.clear()
        try:
            yield
        finally:
            assert normalized(opened) == normalized(expected_opened)
    return (fs, assert_opens)