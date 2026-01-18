import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_list_instance_action_with_marker_and_limit(self):
    server = self._create_server()
    server.stop()
    output = self.nova('instance-action-list %s --limit 1' % server.id)
    marker_req = self._get_column_value_from_single_row_table(output, 'Request_ID')
    action = self._get_list_of_values_from_single_column_table(output, 'Action')
    self.assertEqual(action, ['stop'])
    output = self.nova('instance-action-list %s --limit 1 --marker %s' % (server.id, marker_req))
    action = self._get_list_of_values_from_single_column_table(output, 'Action')
    self.assertEqual(action, ['create'])
    if not self.expect_event_hostId_field:
        output = self.nova('instance-action %s %s' % (server.id, marker_req))
        self.assertNotIn("'host'", output)
        self.assertNotIn("'hostId'", output)