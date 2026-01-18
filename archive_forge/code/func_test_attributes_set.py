import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
def test_attributes_set(self):

    class ReferenceTest(self.Implementation):
        scenarios = [('1', {'foo': 1, 'bar': 2}), ('2', {'foo': 2, 'bar': 4})]

        def test_check_foo(self):
            self.assertEqual(self.foo * 2, self.bar)
    test = ReferenceTest('test_check_foo')
    log = []
    result = LoggingResult(log)
    test.run(result)
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(2, result.testsRun)