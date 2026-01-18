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
@unittest.skipIf(sys.platform.startswith('win'), 'This test is only appropriate for POSIX-like systems.')
@unittest.skipIf(is_emscripten or is_wasi, "Emscripten's/WASI's umask is a stub.")
def test_mkpath_with_custom_mode(self):
    umask = os.umask(2)
    os.umask(umask)
    mkpath(self.target, 448)
    self.assertEqual(stat.S_IMODE(os.stat(self.target).st_mode), 448 & ~umask)
    mkpath(self.target2, 365)
    self.assertEqual(stat.S_IMODE(os.stat(self.target2).st_mode), 365 & ~umask)