import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def catchLogs(testCase, logPublisher=globalLogPublisher):
    """
    Catch the global log stream.

    @param testCase: The test case to add a cleanup to.

    @param logPublisher: the log publisher to add and remove observers for.

    @return: a 0-argument callable that returns a list of textual log messages
        for comparison.
    @rtype: L{list} of L{unicode}
    """
    logs = []
    logPublisher.addObserver(logs.append)
    testCase.addCleanup(lambda: logPublisher.removeObserver(logs.append))
    return lambda: [formatEvent(event) for event in logs]