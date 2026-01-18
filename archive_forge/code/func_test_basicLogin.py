from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def test_basicLogin(self):
    """
        Calling C{login} on a portal with correct credentials and an interface
        that the portal's realm supports works.
        """
    login = self.successResultOf(self.portal.login(credentials.UsernamePassword(b'bob', b'hello'), self, ITestable))
    iface, impl, logout = login
    self.assertEqual(iface, ITestable)
    self.assertTrue(iface.providedBy(impl), f'{impl} does not implement {iface}')
    self.assertTrue(impl.original.loggedIn)
    self.assertTrue(not impl.original.loggedOut)
    logout()
    self.assertTrue(impl.original.loggedOut)