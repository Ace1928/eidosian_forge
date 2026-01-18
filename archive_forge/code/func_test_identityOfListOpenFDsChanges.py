import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_identityOfListOpenFDsChanges(self):
    """
        Check that the identity of _listOpenFDs changes after running
        _listOpenFDs the first time, but not after the second time it's run.

        In other words, check that the monkey patching actually works.
        """
    detector = process._FDDetector()
    first = detector._listOpenFDs.__name__
    detector._listOpenFDs()
    second = detector._listOpenFDs.__name__
    detector._listOpenFDs()
    third = detector._listOpenFDs.__name__
    self.assertNotEqual(first, second)
    self.assertEqual(second, third)