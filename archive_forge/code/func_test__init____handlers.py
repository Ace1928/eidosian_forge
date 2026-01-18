from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__init____handlers(self):
    handlers = [('quux', 'baz')]
    run = RunTest('bar', handlers)
    self.assertEqual(handlers, run.handlers)