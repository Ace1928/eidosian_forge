from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_modePreservation(self) -> None:
    """
        Check rotated files have same permissions as original.
        """
    open(self.path, 'w').close()
    os.chmod(self.path, 455)
    mode = os.stat(self.path)[stat.ST_MODE]
    log = logfile.LogFile(self.name, self.dir)
    self.addCleanup(log.close)
    log.write('abc')
    log.rotate()
    self.assertEqual(mode, os.stat(self.path)[stat.ST_MODE])