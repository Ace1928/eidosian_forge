import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_deprecatedUnjellyingInstanceAtom(self):
    """
        Unjellying the instance atom is deprecated with 15.0.0.
        """
    jelly.unjelly(['instance', ['class', 'twisted.spread.test.test_jelly.A'], ['dictionary']])
    warnings = self.flushWarnings()
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], 'Unjelly support for the instance atom is deprecated since Twisted 15.0.0.  Upgrade peer for modern instance support.')
    self.assertEqual(warnings[0]['category'], DeprecationWarning)