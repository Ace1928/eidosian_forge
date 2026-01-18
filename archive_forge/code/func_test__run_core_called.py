from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_core_called(self):
    case = self.make_case()
    log = []
    run = RunTest(case, lambda x: x)
    run._run_core = lambda: log.append('foo')
    run.run()
    self.assertEqual(['foo'], log)