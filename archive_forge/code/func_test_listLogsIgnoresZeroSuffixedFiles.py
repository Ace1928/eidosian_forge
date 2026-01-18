from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_listLogsIgnoresZeroSuffixedFiles(self) -> None:
    """
        L{LogFile.listLogs} ignores log files which rotated suffix is 0.
        """
    log = logfile.LogFile(self.name, self.dir)
    self.addCleanup(log.close)
    for i in range(0, 3):
        with open(f'{log.path}.{i}', 'w') as fp:
            fp.write('123')
    self.assertEqual([1, 2], log.listLogs())