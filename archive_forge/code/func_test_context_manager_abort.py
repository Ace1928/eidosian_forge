import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def test_context_manager_abort(self):

    def abort():
        with atomicfile.AtomicFile('test') as f:
            f.write(b'foo\n')
            raise AssertionError
    self.assertRaises(AssertionError, abort)
    self.assertEqual([], os.listdir('.'))