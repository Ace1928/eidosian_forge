from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_endpoint_list_filter(self):
    endpoint_id = self._create_dummy_endpoint(add_clean_up=False)
    project_id = self._create_dummy_project(add_clean_up=False)
    raw_output = self.openstack('endpoint add project %(endpoint_id)s %(project_id)s' % {'project_id': project_id, 'endpoint_id': endpoint_id})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('endpoint list --endpoint %s' % endpoint_id)
    self.assertIn(project_id, raw_output)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.ENDPOINT_LIST_PROJECT_HEADERS)
    raw_output = self.openstack('endpoint list --project %s' % project_id)
    self.assertIn(endpoint_id, raw_output)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.ENDPOINT_LIST_HEADERS)