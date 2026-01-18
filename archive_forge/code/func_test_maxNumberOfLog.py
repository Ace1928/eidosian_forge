from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_maxNumberOfLog(self) -> None:
    """
        Test it respect the limit on the number of files when maxRotatedFiles
        is not None.
        """
    log = logfile.LogFile(self.name, self.dir, rotateLength=10, maxRotatedFiles=3)
    self.addCleanup(log.close)
    log.write('1' * 11)
    log.write('2' * 11)
    self.assertTrue(os.path.exists(f'{self.path}.1'))
    log.write('3' * 11)
    self.assertTrue(os.path.exists(f'{self.path}.2'))
    log.write('4' * 11)
    self.assertTrue(os.path.exists(f'{self.path}.3'))
    with open(f'{self.path}.3') as fp:
        self.assertEqual(fp.read(), '1' * 11)
    log.write('5' * 11)
    with open(f'{self.path}.3') as fp:
        self.assertEqual(fp.read(), '2' * 11)
    self.assertFalse(os.path.exists(f'{self.path}.4'))