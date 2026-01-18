from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import os_service_types
from os_service_types.tests import base
def test_is_match(self):
    self.assertEqual(self.is_match, self.service_types.is_match(self.requested, self.found))