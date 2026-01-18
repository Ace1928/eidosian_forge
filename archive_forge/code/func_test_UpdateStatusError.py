from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_UpdateStatusError(self):
    update_status_error = exceptions.UpdateStatusError(fault_string=self.user_fault_string, status_object=self.fault_object, status_object_id=self.fault_object_id, status_record=self.fault_record)
    self.assertEqual(self.user_fault_string, update_status_error.fault_string)
    self.assertEqual(self.fault_object, update_status_error.status_object)
    self.assertEqual(self.fault_object_id, update_status_error.status_object_id)
    self.assertEqual(self.fault_record, update_status_error.status_record)
    self.assertIsInstance(update_status_error, Exception)