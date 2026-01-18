import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def setup_import_issue_package_tests(self, vfs):
    self.addCleanup(setattr, os, 'listdir', os.listdir)
    self.addCleanup(setattr, os.path, 'isfile', os.path.isfile)
    self.addCleanup(setattr, os.path, 'isdir', os.path.isdir)
    self.addCleanup(sys.path.__setitem__, slice(None), list(sys.path))

    def list_dir(path):
        return list(vfs[path])
    os.listdir = list_dir
    os.path.isdir = lambda path: not path.endswith('.py')
    os.path.isfile = lambda path: path.endswith('.py')