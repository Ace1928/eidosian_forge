import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
def test_join_path_common_subdir(self):
    data01 = pathlib.Path(__file__).parent.joinpath('data01')
    data02 = pathlib.Path(__file__).parent.joinpath('data02')
    prefix = str(data01.parent)
    path = MultiplexedPath(data01, data02)
    self.assertIsInstance(path.joinpath('subdirectory'), MultiplexedPath)
    self.assertEqual(str(path.joinpath('subdirectory', 'subsubdir'))[len(prefix) + 1:], os.path.join('data02', 'subdirectory', 'subsubdir'))