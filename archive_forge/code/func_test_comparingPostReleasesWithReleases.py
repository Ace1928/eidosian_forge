from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingPostReleasesWithReleases(self):
    """
        Post releases are always greater than versions without post
        releases.
        """
    va = Version('whatever', 1, 0, 0, post=1)
    vb = Version('whatever', 1, 0, 0)
    self.assertTrue(va > vb)
    self.assertFalse(va < vb)
    self.assertNotEquals(vb, va)