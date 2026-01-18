import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_is_dir(self):
    self.assertEqual(MultiplexedPath(self.folder).is_dir(), True)