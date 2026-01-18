from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_prepared_result_uncaught_Exception_raised(self):
    e = KeyError('Yo')

    class Case(TestCase):

        def test(self):
            raise e
    case = Case('test')
    log = []

    def log_exc(self, result, err):
        log.append((result, err))
    run = RunTest(case, [(ValueError, log_exc)])
    run.result = ExtendedTestResult()
    self.assertThat(lambda: run._run_prepared_result(run.result), Raises(MatchesException(KeyError)))
    self.assertEqual([('startTest', case), ('stopTest', case)], run.result._events)
    self.assertEqual([], log)