import abc
import importlib
import io
import sys
import types
import pathlib
import contextlib
from . import data01
from ..abc import ResourceReader
from .compat.py39 import import_helper, os_helper
from . import zip as zip_
from importlib.machinery import ModuleSpec
def test_missing_path(self):
    """
        Attempting to open or read or request the path for a
        non-existent path should succeed if open_resource
        can return a viable data stream.
        """
    bytes_data = io.BytesIO(b'Hello, world!')
    package = create_package(file=bytes_data, path=FileNotFoundError())
    self.execute(package, 'utf-8.file')
    self.assertEqual(package.__loader__._path, 'utf-8.file')