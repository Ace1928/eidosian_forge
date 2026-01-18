from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_getLog(self) -> None:
    """
        Test retrieving log files with L{DailyLogFile.getLog}.
        """
    data = ['1\n', '2\n', '3\n']
    log = RiggedDailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    for d in data:
        log.write(d)
    log.flush()
    r = log.getLog(0.0)
    self.addCleanup(r.close)
    self.assertEqual(data, r.readLines())
    self.assertRaises(ValueError, log.getLog, 86400)
    log._clock = 86401
    r.close()
    log.rotate()
    r = log.getLog(0)
    self.addCleanup(r.close)
    self.assertEqual(data, r.readLines())