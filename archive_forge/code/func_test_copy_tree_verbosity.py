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
def test_copy_tree_verbosity(self):
    mkpath(self.target, verbose=0)
    copy_tree(self.target, self.target2, verbose=0)
    self.assertEqual(self._logs, [])
    remove_tree(self.root_target, verbose=0)
    mkpath(self.target, verbose=0)
    a_file = os.path.join(self.target, 'ok.txt')
    with open(a_file, 'w') as f:
        f.write('some content')
    wanted = ['copying %s -> %s' % (a_file, self.target2)]
    copy_tree(self.target, self.target2, verbose=1)
    self.assertEqual(self._logs, wanted)
    remove_tree(self.root_target, verbose=0)
    remove_tree(self.target2, verbose=0)