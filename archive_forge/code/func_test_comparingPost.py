from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_comparingPost(self):
    """
        The value specified as the postrelease is used in version comparisons.
        """
    va = Version('whatever', 1, 0, 0, post=1)
    vb = Version('whatever', 1, 0, 0, post=2)
    self.assertTrue(va < vb)
    self.assertTrue(vb > va)
    self.assertTrue(va <= vb)
    self.assertTrue(vb >= va)
    self.assertTrue(va != vb)
    self.assertTrue(vb == Version('whatever', 1, 0, 0, post=2))
    self.assertTrue(va == va)