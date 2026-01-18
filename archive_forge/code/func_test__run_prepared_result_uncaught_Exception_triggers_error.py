from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_prepared_result_uncaught_Exception_triggers_error(self):
    e = SystemExit(0)

    class Case(TestCase):

        def test(self):
            raise e
    case = Case('test')
    log = []

    def log_exc(self, result, err):
        log.append((result, err))
    run = RunTest(case, [], log_exc)
    run.result = ExtendedTestResult()
    self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(SystemExit)))
    self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
    self.assertEqual([(run.result, e)], log)