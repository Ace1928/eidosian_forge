from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def mock_localtime(*args: object) -> list[int]:
    self.assertEqual((), args)
    return list(range(0, 9))