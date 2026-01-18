from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def test_extended_decorator_for_run_test(self):
    marker = object()

    class FooRunTest(RunTest):

        def __init__(self, case, handlers=None, bar=None):
            super().__init__(case, handlers)
            self.bar = bar

        def run(self, result=None):
            return self.bar

    class SomeCase(TestCase):

        @run_test_with(FooRunTest, bar=marker)
        def test_foo(self):
            pass
    result = TestResult()
    case = SomeCase('test_foo')
    from_run_test = case.run(result)
    self.assertThat(from_run_test, Is(marker))