from __future__ import division, absolute_import
import sys
import unittest
import operator
from incremental import getVersionString, IncomparableVersions
from incremental import Version, _inf
from twisted.trial.unittest import TestCase
def test_baseWithNEXT(self):
    """
        The C{base} method returns just "NEXT" when NEXT is the major version.
        """
    self.assertEqual(Version('foo', 'NEXT', 0, 0).base(), 'NEXT')