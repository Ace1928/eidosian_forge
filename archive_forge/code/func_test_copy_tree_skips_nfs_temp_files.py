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
def test_copy_tree_skips_nfs_temp_files(self):
    mkpath(self.target, verbose=0)
    a_file = os.path.join(self.target, 'ok.txt')
    nfs_file = os.path.join(self.target, '.nfs123abc')
    for f in (a_file, nfs_file):
        with open(f, 'w') as fh:
            fh.write('some content')
    copy_tree(self.target, self.target2)
    self.assertEqual(os.listdir(self.target2), ['ok.txt'])
    remove_tree(self.root_target, verbose=0)
    remove_tree(self.target2, verbose=0)