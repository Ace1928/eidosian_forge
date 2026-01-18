import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def registerConsumer(self, cons):
    assert self._cons is None
    self._cons = cons
    self._cons.registerProducer(self, True)
    for chunk in self._buffer:
        self._conswrite(chunk)
    self._buffer = None
    if self.isConnected:
        self._onConnLost = d = defer.Deferred()
        d.addBoth(self._unregConsumer)
        return d
    else:
        self._cons.unregisterProducer()
        self._cons = None
        return defer.succeed(None)