import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_countTestCases_2_scenarios(self):

    class ReferenceTest(self.Implementation):
        scenarios = [('1', {'foo': 1, 'bar': 2}), ('2', {'foo': 2, 'bar': 4})]

        def test_check_foo(self):
            pass
    test = ReferenceTest('test_check_foo')
    self.assertEqual(2, test.countTestCases())