import base64
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.cred import error, portal
from twisted.cred.checkers import (
from twisted.cred.credentials import IUsernamePassword
from twisted.internet.address import IPv4Address
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial import unittest
from twisted.web._auth import basic, digest
from twisted.web._auth.basic import BasicCredentialFactory
from twisted.web._auth.wrapper import HTTPAuthSessionWrapper, UnauthorizedResource
from twisted.web.iweb import ICredentialFactory
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.server import NOT_DONE_YET
from twisted.web.static import Data
from twisted.web.test.test_web import DummyRequest
def test_incorrectPadding(self):
    """
        L{basic.BasicCredentialFactory.decode} decodes a base64-encoded
        response with incorrect padding.
        """
    response = b64encode(b''.join([self.username, b':', self.password]))
    response = response.strip(b'=')
    creds = self.credentialFactory.decode(response, self.request)
    self.assertTrue(verifyObject(IUsernamePassword, creds))
    self.assertTrue(creds.checkPassword(self.password))