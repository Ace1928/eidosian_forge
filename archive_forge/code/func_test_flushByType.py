import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_flushByType(self):
    """
        Check that flushing the observer remove all failures of the given type.
        """
    self.test_error()
    f = self._makeRuntimeFailure()
    self.observer.gotEvent(dict(message=(), time=time.time(), isError=1, system='-', failure=f, why=None))
    flushed = self.observer.flushErrors(ZeroDivisionError)
    self.assertEqual(self.observer.getErrors(), [f])
    self.assertEqual(len(flushed), 1)
    self.assertTrue(flushed[0].check(ZeroDivisionError))