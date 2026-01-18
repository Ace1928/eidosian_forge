from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def getGoodCredentials(self):
    for u, p in self._validCredentials:
        yield (self.credClass(u, self.networkHash(p)), u)