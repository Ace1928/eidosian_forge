from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithDev(self):
    """
        The base version includes 'devX' for versions with dev releases.
        """
    self.assertEqual(Version('foo', 1, 0, 0, dev=8).base(), '1.0.0.dev8')