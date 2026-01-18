from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def test_multiply_tests_by_their_scenarios(self):
    loader = TestLoader()
    suite = loader.suiteClass()
    test_instance = PretendVaryingTest('test_nothing')
    multiply_tests_by_their_scenarios(test_instance, suite)
    self.assertEqual(['a', 'a', 'b', 'b'], get_generated_test_attributes(suite, 'value'))