from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_prepared_result_calls_start_and_stop_test(self):
    result = ExtendedTestResult()
    case = self.make_case()
    run = RunTest(case, lambda x: x)
    run.run(result)
    self.assertEqual([('startTest', case), ('addSuccess', case), ('stopTest', case)], result._events)