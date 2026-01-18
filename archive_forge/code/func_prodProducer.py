import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
def prodProducer(self):
    """
        Repeatedly prod a non-streaming producer to produce data.
        """
    self.openCall = None
    if self.myproducer is not None:
        self.openCall = reactor.callLater(0, self.prodProducer)
        self.myproducer.resumeProducing()