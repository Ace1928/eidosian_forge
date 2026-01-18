from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingNEXTReleases(self):
    """
        NEXT releases are always larger than numbered releases.
        """
    va = Version('whatever', 'NEXT', 0, 0)
    vb = Version('whatever', 1, 0, 0)
    self.assertTrue(va > vb)
    self.assertFalse(va < vb)
    self.assertNotEquals(vb, va)