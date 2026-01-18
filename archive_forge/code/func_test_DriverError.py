from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_DriverError(self):
    driver_error = exceptions.DriverError(user_fault_string=self.user_fault_string, operator_fault_string=self.operator_fault_string)
    self.assertEqual(self.user_fault_string, driver_error.user_fault_string)
    self.assertEqual(self.operator_fault_string, driver_error.operator_fault_string)
    self.assertIsInstance(driver_error, Exception)