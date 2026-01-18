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
def registerAddress(self, domainURL, logicalURL, physicalURL):
    if domainURL.host != self.domain:
        log.msg("Registration for domain we don't handle.")
        return defer.fail(RegistrationError(404))
    if logicalURL.host != self.domain:
        log.msg("Registration for domain we don't handle.")
        return defer.fail(RegistrationError(404))
    if logicalURL.username in self.users:
        dc, old = self.users[logicalURL.username]
        dc.reset(3600)
    else:
        dc = reactor.callLater(3600, self._expireRegistration, logicalURL.username)
    log.msg(f'Registered {logicalURL.toString()} at {physicalURL.toString()}')
    self.users[logicalURL.username] = (dc, physicalURL)
    return defer.succeed(Registration(int(dc.getTime() - time.time()), physicalURL))