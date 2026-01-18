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
def test_has_keyDeprecation(self):
    """
        has_key is now deprecated.
        """
    sut = mail.mail.DomainWithDefaultDict({}, 'Default')
    sut.has_key('anything')
    message = 'twisted.mail.mail.DomainWithDefaultDict.has_key was deprecated in Twisted 16.3.0. Use the `in` keyword instead.'
    warnings = self.flushWarnings([self.test_has_keyDeprecation])
    self.assertEqual(1, len(warnings))
    self.assertEqual(DeprecationWarning, warnings[0]['category'])
    self.assertEqual(message, warnings[0]['message'])