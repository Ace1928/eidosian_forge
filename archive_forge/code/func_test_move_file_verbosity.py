import unittest
import os
import errno
from unittest.mock import patch
from distutils.file_util import move_file, copy_file
from distutils import log
from distutils.tests import support
from distutils.errors import DistutilsFileError
from test.support.os_helper import unlink
def test_move_file_verbosity(self):
    f = open(self.source, 'w')
    try:
        f.write('some content')
    finally:
        f.close()
    move_file(self.source, self.target, verbose=0)
    wanted = []
    self.assertEqual(self._logs, wanted)
    move_file(self.target, self.source, verbose=0)
    move_file(self.source, self.target, verbose=1)
    wanted = ['moving %s -> %s' % (self.source, self.target)]
    self.assertEqual(self._logs, wanted)
    move_file(self.target, self.source, verbose=0)
    self._logs = []
    os.mkdir(self.target_dir)
    move_file(self.source, self.target_dir, verbose=1)
    wanted = ['moving %s -> %s' % (self.source, self.target_dir)]
    self.assertEqual(self._logs, wanted)