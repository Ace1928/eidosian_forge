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
@skipIf(sys.version_info >= (3,), 'not ported to Python 3')
def test_cyclicAlias(self):
    """
        Check that a cycle in alias resolution is correctly handled.
        """
    aliases = {}
    domain = {'': TestDomain(aliases, [])}
    A1 = mail.alias.AddressAlias('alias2', domain, 'alias1')
    A2 = mail.alias.AddressAlias('alias3', domain, 'alias2')
    A3 = mail.alias.AddressAlias('alias1', domain, 'alias3')
    aliases.update({'alias1': A1, 'alias2': A2, 'alias3': A3})
    self.assertEqual(aliases['alias1'].resolve(aliases), None)
    self.assertEqual(aliases['alias2'].resolve(aliases), None)
    self.assertEqual(aliases['alias3'].resolve(aliases), None)
    A4 = MockAliasGroup(['|echo', 'alias1'], domain, 'alias4')
    aliases['alias4'] = A4
    res = A4.resolve(aliases)
    r = map(str, res.objs)
    r.sort()
    expected = map(str, [mail.alias.MessageWrapper(DummyProcess(), 'echo')])
    expected.sort()
    self.assertEqual(r, expected)