import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def mockDTPInstance(result):
    """
            Fake an incoming connection and create a mock DTPInstance so
            that PORT command will start processing the request.

            @param result: parameter used in L{Deferred}
            """
    self.serverProtocol.dtpFactory.deferred.callback(None)
    self.serverProtocol.dtpInstance = object()
    self.serverProtocol.shell.openForWriting = failingOpenForWriting
    return result