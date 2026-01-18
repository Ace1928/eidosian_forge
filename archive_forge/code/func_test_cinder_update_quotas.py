from openstack import exceptions
from openstack.network.v2 import quota as _quota
from openstack.tests.unit import base
def test_cinder_update_quotas(self):
    project = self._get_project_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url('identity', 'public', append=['v3', 'projects', project.project_id]), json={'project': project.json_response['project']}), self.get_cinder_discovery_mock_dict(), dict(method='PUT', uri=self.get_mock_url('volumev3', 'public', append=['os-quota-sets', project.project_id]), json=dict(quota_set={'volumes': 1}), validate=dict(json={'quota_set': {'volumes': 1}}))])
    self.cloud.set_volume_quotas(project.project_id, volumes=1)
    self.assert_calls()