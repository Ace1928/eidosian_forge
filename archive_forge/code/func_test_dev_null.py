import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def test_dev_null(self):
    if sys.platform == 'win32':
        brz_log = 'NUL'
    else:
        brz_log = '/dev/null'
    self.overrideEnv('BRZ_LOG', brz_log)
    self.assertPathDoesNotExist(self.default_log())
    out = self.run_brz_subprocess('version')[0]
    self.assertTrue(len(out) > 0)
    self.assertContainsRe(out, b'(?m)^  Breezy log file: ' + brz_log.encode('ascii'))
    self.assertPathDoesNotExist(self.default_log())