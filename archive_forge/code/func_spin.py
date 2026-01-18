import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def spin():
    for value in count:
        if value == howMany:
            connection.loseConnection()
            return
        connection.write(b'%d' % (value,))
        break
    reactor.callLater(0, spin)