import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorThreads(reactor, None), 'Nothing to wake up for without thread support')
def testWakeUp(self):
    d = Deferred()

    def wake():
        time.sleep(0.1)
        reactor.callFromThread(d.callback, None)
    reactor.callInThread(wake)
    return d