from octavia_lib.api.drivers import exceptions
from octavia_lib.tests.unit import base
def test_UpdateStatisticsError(self):
    update_stats_error = exceptions.UpdateStatisticsError(fault_string=self.user_fault_string, stats_object=self.fault_object, stats_object_id=self.fault_object_id, stats_record=self.fault_record)
    self.assertEqual(self.user_fault_string, update_stats_error.fault_string)
    self.assertEqual(self.fault_object, update_stats_error.stats_object)
    self.assertEqual(self.fault_object_id, update_stats_error.stats_object_id)
    self.assertEqual(self.fault_record, update_stats_error.stats_record)
    self.assertIsInstance(update_stats_error, Exception)