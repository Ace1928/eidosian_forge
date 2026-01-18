from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_nonExistentDir(self) -> None:
    """
        Specifying an invalid directory to L{LogFile} raises C{IOError}.
        """
    e = self.assertRaises(IOError, logfile.LogFile, self.name, 'this_dir_does_not_exist')
    self.assertEqual(e.errno, errno.ENOENT)