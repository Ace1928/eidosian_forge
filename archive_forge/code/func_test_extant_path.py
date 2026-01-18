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
def test_extant_path(self):
    bytes_data = io.BytesIO(b'Hello, world!')
    path = __file__
    package = create_package(file=bytes_data, path=path)
    self.execute(package, 'utf-8.file')
    self.assertEqual(package.__loader__._path, 'utf-8.file')