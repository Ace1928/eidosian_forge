from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test___init___short(self):
    run = RunTest('bar')
    self.assertEqual('bar', run.case)
    self.assertEqual([], run.handlers)