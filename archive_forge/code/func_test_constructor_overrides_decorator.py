from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_constructor_overrides_decorator(self):
    marker = object()

    class DifferentRunTest(RunTest):

        def run(self, result=None):
            return marker

    class SomeCase(TestCase):

        @run_test_with(CustomRunTest)
        def test_foo(self):
            pass
    result = TestResult()
    case = SomeCase('test_foo', runTest=DifferentRunTest)
    from_run_test = case.run(result)
    self.assertThat(from_run_test, Is(marker))