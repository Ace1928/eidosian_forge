from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_cinder_get_quotas(self):
    project = self._get_project_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url('identity', 'public', append=['v3', 'projects', project.project_id]), json={'project': project.json_response['project']}), self.get_cinder_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['os-quota-sets', project.project_id], qs_elements=['usage=False']), json=dict(quota_set={'snapshots': 10, 'volumes': 20}))])
    self.cloud.get_volume_quotas(project.project_id)
    self.assert_calls()