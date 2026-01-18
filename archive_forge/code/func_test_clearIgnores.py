import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_clearIgnores(self):
    """
        Check that C{_clearIgnores} ensures that previously ignored errors
        get captured.
        """
    self.observer._ignoreErrors(ZeroDivisionError)
    self.observer._clearIgnores()
    f = makeFailure()
    self.observer.gotEvent({'message': (), 'time': time.time(), 'isError': 1, 'system': '-', 'failure': f, 'why': None})
    self.assertEqual(self.observer.getErrors(), [f])