from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_endpoint_add_remove_project(self):
    endpoint_id = self._create_dummy_endpoint(add_clean_up=False)
    project_id = self._create_dummy_project(add_clean_up=False)
    raw_output = self.openstack('endpoint add project %(endpoint_id)s %(project_id)s' % {'project_id': project_id, 'endpoint_id': endpoint_id})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('endpoint remove project %(endpoint_id)s %(project_id)s' % {'project_id': project_id, 'endpoint_id': endpoint_id})
    self.assertEqual(0, len(raw_output))