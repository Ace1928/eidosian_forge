from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test__run_user_uncaught_Exception_from_exception_handler_raised(self):
    case = self.make_case()

    def broken_handler(exc_info):
        raise ValueError('boo')
    case.addOnException(broken_handler)
    e = KeyError('Yo')

    def raises():
        raise e
    log = []

    def log_exc(self, result, err):
        log.append((result, err))
    run = RunTest(case, [(ValueError, log_exc)])
    run.result = ExtendedTestResult()
    self.assertThat(lambda: run._run_user(raises), Raises(MatchesException(ValueError)))
    self.assertEqual([], run.result._events)
    self.assertEqual([], log)