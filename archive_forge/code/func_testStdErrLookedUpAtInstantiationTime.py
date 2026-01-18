import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testStdErrLookedUpAtInstantiationTime(self):
    old_stderr = sys.stderr
    f = io.StringIO()
    sys.stderr = f
    try:
        runner = unittest.TextTestRunner()
        self.assertTrue(runner.stream.stream is f)
    finally:
        sys.stderr = old_stderr