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
def test_fallbackForAllBadExchanges(self):
    """
        L{MXCalculator.getMX} returns the MX record with the lowest preference
        if all the MX records in the response have been marked bad.
        """
    domain = 'example.com'
    bad = 'bad.example.com'
    worse = 'worse.example.com'
    records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, bad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, worse))]
    self.mx.markBad(bad)
    self.mx.markBad(worse)
    return self._exchangeTest(domain, records, bad)