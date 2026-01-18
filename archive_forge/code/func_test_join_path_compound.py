import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_join_path_compound(self):
    path = MultiplexedPath(self.folder)
    assert not path.joinpath('imaginary/foo.py').exists()