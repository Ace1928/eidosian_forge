import unittest
import os
import errno
from unittest.mock import patch
from distutils.file_util import move_file, copy_file
from distutils import log
from distutils.tests import support
from distutils.errors import DistutilsFileError
from test.support.os_helper import unlink
def test_move_file_exception_unpacking_unlink(self):
    with patch('os.rename', side_effect=OSError(errno.EXDEV, 'wrong')), patch('os.unlink', side_effect=OSError('wrong', 1)), self.assertRaises(DistutilsFileError):
        with open(self.source, 'w') as fobj:
            fobj.write('spam eggs')
        move_file(self.source, self.target, verbose=0)