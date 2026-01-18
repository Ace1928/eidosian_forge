import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
def test_ignoreErrors(self):
    """
        Check that C{_ignoreErrors} actually causes errors to be ignored.
        """
    self.observer._ignoreErrors(ZeroDivisionError)
    f = makeFailure()
    self.observer.gotEvent({'message': (), 'time': time.time(), 'isError': 1, 'system': '-', 'failure': f, 'why': None})
    self.assertEqual(self.observer.getErrors(), [])