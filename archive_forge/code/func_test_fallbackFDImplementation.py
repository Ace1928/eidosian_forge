import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_fallbackFDImplementation(self):
    """
        L{_FDDetector._fallbackFDImplementation}, the implementation of last
        resort, succeeds with a fixed range of integers from 0 to 1024 when the
        L{resource} module is not importable.
        """
    self.hideResourceModule()
    self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))