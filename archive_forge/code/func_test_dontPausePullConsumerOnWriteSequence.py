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
def test_dontPausePullConsumerOnWriteSequence(self):
    """
        Like L{test_dontPausePullConsumerOnWrite}, but for a call to
        C{writeSequence} rather than L{IConsumer.write}.

        C{writeSequence} is not part of L{IConsumer}, but
        L{abstract.FileDescriptor} has supported consumery behavior in response
        to calls to L{writeSequence} forever.
        """
    return self._dontPausePullConsumerTest('writeSequence')