from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_defaultPermissions(self) -> None:
    """
        Test the default permission of the log file: if the file exist, it
        should keep the permission.
        """
    with open(self.path, 'wb'):
        os.chmod(self.path, 455)
        currentMode = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
    log1 = logfile.LogFile(self.name, self.dir)
    self.assertEqual(stat.S_IMODE(os.stat(self.path)[stat.ST_MODE]), currentMode)
    self.addCleanup(log1.close)