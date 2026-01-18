import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_init_no_paths(self):
    with self.assertRaises(FileNotFoundError):
        MultiplexedPath()