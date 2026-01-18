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
def test_getChildWithDefaultAuthorized(self):
    """
        Resource traversal which encounters an L{HTTPAuthSessionWrapper}
        results in an L{IResource} which renders the L{IResource} avatar
        retrieved from the portal when the request has a valid I{Authorization}
        header.
        """
    self.credentialFactories.append(BasicCredentialFactory('example.com'))
    request = self.makeRequest([self.childName])
    child = self._authorizedBasicLogin(request)
    d = request.notifyFinish()

    def cbFinished(ignored):
        self.assertEqual(request.written, [self.childContent])
    d.addCallback(cbFinished)
    request.render(child)
    return d