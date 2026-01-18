from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_UnsupportedOptionError(self):
    unsupported_option_error = exceptions.UnsupportedOptionError(user_fault_string=self.user_fault_string, operator_fault_string=self.operator_fault_string)
    self.assertEqual(self.user_fault_string, unsupported_option_error.user_fault_string)
    self.assertEqual(self.operator_fault_string, unsupported_option_error.operator_fault_string)
    self.assertIsInstance(unsupported_option_error, Exception)