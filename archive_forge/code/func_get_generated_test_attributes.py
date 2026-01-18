from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def get_generated_test_attributes(suite, attr_name):
    """Return the `attr_name` attribute from all tests in the suite"""
    return sorted([getattr(t, attr_name) for t in iter_suite_tests(suite)])