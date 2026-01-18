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
def testRelayDelivery(self):
    insServ = mail.mail.MailService()
    insServ.smtpPortal.registerChecker(cred.checkers.AllowAnonymousAccess())
    domain = mail.maildir.MaildirDirdbmDomain(insServ, 'insertionDomain')
    insServ.addDomain('insertion.domain', domain)
    os.mkdir('insertionQueue')
    insServ.setQueue(mail.relaymanager.Queue('insertionQueue'))
    insServ.domains.setDefaultDomain(mail.relay.DomainQueuer(insServ))
    manager = mail.relaymanager.SmartHostSMTPRelayingManager(insServ.queue)
    manager.fArgs += ('test.identity.hostname',)
    helper = mail.relaymanager.RelayStateHelper(manager, 1)
    manager.mxcalc = mail.relaymanager.MXCalculator(self.resolver)
    self.auth.addresses['destination.domain'] = ['127.0.0.1']
    f = insServ.getSMTPFactory()
    self.insServer = reactor.listenTCP(0, f, interface='127.0.0.1')
    destServ = mail.mail.MailService()
    destServ.smtpPortal.registerChecker(cred.checkers.AllowAnonymousAccess())
    domain = mail.maildir.MaildirDirdbmDomain(destServ, 'destinationDomain')
    domain.addUser('user', 'password')
    destServ.addDomain('destination.domain', domain)
    os.mkdir('destinationQueue')
    destServ.setQueue(mail.relaymanager.Queue('destinationQueue'))
    helper = mail.relaymanager.RelayStateHelper(manager, 1)
    helper.startService()
    f = destServ.getSMTPFactory()
    self.destServer = reactor.listenTCP(0, f, interface='127.0.0.1')
    manager.PORT = self.destServer.getHost().port
    client = LineSendingProtocol(['HELO meson', 'MAIL FROM: <user@wherever>', 'RCPT TO: <user@destination.domain>', 'DATA', 'This is the message', '.', 'QUIT'])
    done = Deferred()
    f = protocol.ClientFactory()
    f.protocol = lambda: client
    f.clientConnectionLost = lambda *args: done.callback(None)
    reactor.connectTCP('127.0.0.1', self.insServer.getHost().port, f)

    def finished(ign):
        delivery = manager.checkState()

        def delivered(ign):
            mbox = domain.requestAvatar('user', None, pop3.IMailbox)[1]
            msg = mbox.getMessage(0).read()
            self.assertNotEqual(msg.find('This is the message'), -1)
            self.insServer.stopListening()
            self.destServer.stopListening()
            helper.stopService()
        delivery.addCallback(delivered)
        return delivery
    done.addCallback(finished)
    return done