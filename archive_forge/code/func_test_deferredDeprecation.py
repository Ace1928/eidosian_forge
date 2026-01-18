from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_deferredDeprecation(self):
    """
        L{LoopingCall.deferred} is deprecated.
        """
    loop = task.LoopingCall(lambda: None)
    loop.deferred
    message = 'twisted.internet.task.LoopingCall.deferred was deprecated in Twisted 16.0.0; please use the deferred returned by start() instead'
    warnings = self.flushWarnings([self.test_deferredDeprecation])
    self.assertEqual(1, len(warnings))
    self.assertEqual(DeprecationWarning, warnings[0]['category'])
    self.assertEqual(message, warnings[0]['message'])