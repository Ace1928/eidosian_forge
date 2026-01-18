from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_getVersionStringWithPost(self):
    """
        L{getVersionString} includes the postrelease, if any.
        """
    self.assertEqual(getVersionString(Version('whatever', 8, 0, 0, post=1)), 'whatever 8.0.0.post1')