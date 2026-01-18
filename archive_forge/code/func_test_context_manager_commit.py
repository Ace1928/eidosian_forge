import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def test_context_manager_commit(self):
    with atomicfile.AtomicFile('test') as f:
        self.assertPathDoesNotExist('test')
        f.write(b'foo\n')
    self.assertEqual(['test'], os.listdir('.'))
    self.check_file_contents('test', b'foo\n')