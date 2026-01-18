import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_role_group_name_domain_exists(self):
    uris = self._get_mock_role_query_urls(self.role_data, domain_data=self.domain_data, group_data=self.group_data, use_role_name=True, use_group_name=True)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='domains', append=[self.domain_data.domain_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=204)])
    self.register_uris(uris)
    self.assertFalse(self.cloud.grant_role(self.role_data.role_name, group=self.group_data.group_name, domain=self.domain_data.domain_id))
    self.assert_calls()