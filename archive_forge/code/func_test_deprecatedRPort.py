from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_deprecatedRPort(self):
    """
        Setting rport to True is deprecated, but still produces a Via header
        with the expected properties.
        """
    v = sip.Via('foo.bar', rport=True)
    warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedRPort])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], 'rport=True is deprecated since Twisted 9.0.')
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
    self.assertTrue(v.rport)
    self.assertTrue(v.rportRequested)
    self.assertIsNone(v.rportValue)