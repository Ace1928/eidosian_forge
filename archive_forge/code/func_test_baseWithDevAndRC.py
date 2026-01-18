from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithDevAndRC(self):
    """
        The base version includes 'rcXdevX' for versions with dev releases and
        a release candidate.
        """
    self.assertEqual(Version('foo', 1, 0, 0, release_candidate=2, dev=8).base(), '1.0.0.rc2.dev8')