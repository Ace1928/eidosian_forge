from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_fromFullPath(self) -> None:
    """
        Test the fromFullPath method.
        """
    log1 = logfile.LogFile(self.name, self.dir, 10, defaultMode=511)
    self.addCleanup(log1.close)
    log2 = logfile.LogFile.fromFullPath(self.path, 10, defaultMode=511)
    self.addCleanup(log2.close)
    self.assertEqual(log1.name, log2.name)
    self.assertEqual(os.path.abspath(log1.path), log2.path)
    self.assertEqual(log1.rotateLength, log2.rotateLength)
    self.assertEqual(log1.defaultMode, log2.defaultMode)