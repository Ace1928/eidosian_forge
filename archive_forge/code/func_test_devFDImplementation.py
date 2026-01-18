import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_devFDImplementation(self):
    """
        L{_FDDetector._devFDImplementation} raises L{OSError} if there is no
        I{/dev/fd} directory, otherwise it returns the basenames of its children
        interpreted as integers.
        """
    self.devfs = False
    self.assertRaises(OSError, self.detector._devFDImplementation)
    self.devfs = True
    self.accurateDevFDResults = False
    self.assertEqual([0, 1, 2], self.detector._devFDImplementation())