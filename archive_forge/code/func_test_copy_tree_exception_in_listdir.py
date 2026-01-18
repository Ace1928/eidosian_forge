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
def test_copy_tree_exception_in_listdir(self):
    """
        An exception in listdir should raise a DistutilsFileError
        """
    with patch('os.listdir', side_effect=OSError()), self.assertRaises(errors.DistutilsFileError):
        src = self.tempdirs[-1]
        dir_util.copy_tree(src, None)