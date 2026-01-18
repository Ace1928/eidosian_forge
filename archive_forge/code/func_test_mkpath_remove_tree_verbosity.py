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
def test_mkpath_remove_tree_verbosity(self):
    mkpath(self.target, verbose=0)
    wanted = []
    self.assertEqual(self._logs, wanted)
    remove_tree(self.root_target, verbose=0)
    mkpath(self.target, verbose=1)
    wanted = ['creating %s' % self.root_target, 'creating %s' % self.target]
    self.assertEqual(self._logs, wanted)
    self._logs = []
    remove_tree(self.root_target, verbose=1)
    wanted = ["removing '%s' (and everything under it)" % self.root_target]
    self.assertEqual(self._logs, wanted)