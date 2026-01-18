from io import StringIO
from twisted.python.failure import Failure
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial.reporter import TreeReporter
from twisted.trial.unittest import TestCase
def test_forwardedMethods(self) -> None:
    """
        Calling methods of L{DistReporter} add calls to the running queue of
        the test.
        """
    self.distReporter.startTest(self.test)
    self.distReporter.addFailure(self.test, Failure(Exception('foo')))
    self.distReporter.addError(self.test, Failure(Exception('bar')))
    self.distReporter.addSkip(self.test, 'egg')
    self.distReporter.addUnexpectedSuccess(self.test, 'spam')
    self.distReporter.addExpectedFailure(self.test, Failure(Exception('err')), 'foo')
    self.assertEqual(len(self.distReporter.running[self.test.id()]), 6)