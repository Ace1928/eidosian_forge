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
def test_cnameWithGlueRecords(self):
    """
        If an MX lookup returns a CNAME and the MX record for the CNAME, the
        L{Deferred} returned by L{MXCalculator.getMX} should be called back
        with the name from the MX record without further lookups being
        attempted.
        """
    lookedUp = []
    alias = 'alias.example.com'
    canonical = 'canonical.example.com'
    exchange = 'mail.example.com'

    class DummyResolver:

        def lookupMailExchange(self, domain):
            if domain != alias or lookedUp:
                return ([], [], [])
            return defer.succeed(([RRHeader(name=alias, type=Record_CNAME.TYPE, payload=Record_CNAME(canonical)), RRHeader(name=canonical, type=Record_MX.TYPE, payload=Record_MX(name=exchange))], [], []))
    self.mx.resolver = DummyResolver()
    d = self.mx.getMX(alias)
    d.addCallback(self.assertEqual, Record_MX(name=exchange))
    return d