import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
def test_failureWithSuccessfulFallback(self):
    """
        Test that if the MX record lookup fails, fallback is enabled, and an A
        record is available for the name, then the Deferred returned by
        L{MXCalculator.getMX} ultimately fires with a Record_MX instance which
        gives the address in the A record for the name.
        """

    class DummyResolver:
        """
            Fake resolver which will fail an MX lookup but then succeed a
            getHostByName call.
            """

        def lookupMailExchange(self, domain):
            return defer.fail(DNSNameError())

        def getHostByName(self, domain):
            return defer.succeed('1.2.3.4')
    self.mx.resolver = DummyResolver()
    d = self.mx.getMX('domain')
    d.addCallback(self.assertEqual, Record_MX(name='1.2.3.4'))
    return d