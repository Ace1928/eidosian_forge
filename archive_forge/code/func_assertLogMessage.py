from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def assertLogMessage(testCase, expectedMessages, callable, *args, **kwargs):
    """
    Assert that the callable logs the expected messages when called.

    XXX: Put this somewhere where it can be re-used elsewhere. See #6677.

    @param testCase: The test case controlling the test which triggers the
        logged messages and on which assertions will be called.
    @type testCase: L{unittest.SynchronousTestCase}

    @param expectedMessages: A L{list} of the expected log messages
    @type expectedMessages: L{list}

    @param callable: The function which is expected to produce the
        C{expectedMessages} when called.
    @type callable: L{callable}

    @param args: Positional arguments to be passed to C{callable}.
    @type args: L{list}

    @param kwargs: Keyword arguments to be passed to C{callable}.
    @type kwargs: L{dict}
    """
    loggedMessages = []
    log.addObserver(loggedMessages.append)
    testCase.addCleanup(log.removeObserver, loggedMessages.append)
    callable(*args, **kwargs)
    testCase.assertEqual([m['message'][0] for m in loggedMessages], expectedMessages)