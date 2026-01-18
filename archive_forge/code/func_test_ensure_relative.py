import unittest
import os
import stat
import sys
from unittest.mock import patch
from distutils import dir_util, errors
from distutils.dir_util import (mkpath, remove_tree, create_tree, copy_tree,
from distutils import log
from distutils.tests import support
from test.support import is_emscripten, is_wasi
def test_ensure_relative(self):
    if os.sep == '/':
        self.assertEqual(ensure_relative('/home/foo'), 'home/foo')
        self.assertEqual(ensure_relative('some/path'), 'some/path')
    else:
        self.assertEqual(ensure_relative('c:\\home\\foo'), 'c:home\\foo')
        self.assertEqual(ensure_relative('home\\foo'), 'home\\foo')