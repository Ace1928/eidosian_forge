from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_challengerRaisesException(self):
    """
        When a challenger's
        L{getChallenge<IChallengeResponse.getChallenge>} method raises
        any exception, a C{NO} response is sent.
        """

    @implementer(IChallengeResponse)
    class ValueErrorAuthChallenge:
        message = b'A challenge failure'

        def getChallenge(self):
            raise ValueError(self.message)

        def setResponse(self, response):
            """
                Never called.

                @param response: See L{IChallengeResponse.setResponse}
                """

        def moreChallenges(self):
            """
                Never called.
                """

    @implementer(IClientAuthentication)
    class ValueErrorAuthenticator:

        def getName(self):
            return b'ERROR'

        def challengeResponse(self, secret, chal):
            return b'IGNORED'
    bad = ValueErrorAuthChallenge()
    verifyObject(IChallengeResponse, bad)
    self.server.challengers[b'ERROR'] = ValueErrorAuthChallenge
    self.client.registerAuthenticator(ValueErrorAuthenticator())

    def auth():
        return self.client.authenticate(b'secret')
    d = self.connected.addCallback(strip(auth))
    d.addErrback(self.assertClientFailureMessage, ('Server error: ' + str(ValueErrorAuthChallenge.message)).encode('ascii'))
    d.addCallbacks(self._cbStopClient, self._ebGeneral)
    return defer.gatherResults([d, self.loopback()])