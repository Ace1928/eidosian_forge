from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_decorator_for_run_test(self):

    class SomeCase(TestCase):

        @run_test_with(CustomRunTest)
        def test_foo(self):
            pass
    result = TestResult()
    case = SomeCase('test_foo')
    from_run_test = case.run(result)
    self.assertThat(from_run_test, Is(CustomRunTest.marker))