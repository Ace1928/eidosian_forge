from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def test_multiply_tests_no_scenarios(self):
    """Tests with no scenarios attribute aren't multiplied"""
    suite = TestLoader().suiteClass()
    multiply_tests_by_their_scenarios(self, suite)
    self.assertLength(1, list(iter_suite_tests(suite)))