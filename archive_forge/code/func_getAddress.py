import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def getAddress(self, userURI):
    if userURI.host != self.domain:
        return defer.fail(LookupError('unknown domain'))
    if userURI.username in self.users:
        dc, url = self.users[userURI.username]
        return defer.succeed(url)
    else:
        return defer.fail(LookupError('no such user'))