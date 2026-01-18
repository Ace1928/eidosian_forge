import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_no_role(self):
    uris = self.__get('domain', self.domain_data, 'domain_name', [], use_name=True)
    uris.extend([dict(method='GET', uri=self.get_mock_url(resource='roles', append=[self.role_data.role_name]), status_code=404), dict(method='GET', uri=self.get_mock_url(resource='roles', qs_elements=['name=' + self.role_data.role_name]), status_code=200, json={'roles': []})])
    self.register_uris(uris)
    with testtools.ExpectedException(exceptions.SDKException, 'Role {0} not found'.format(self.role_data.role_name)):
        self.cloud.grant_role(self.role_data.role_name, group=self.group_data.group_name, domain=self.domain_data.domain_name)
    self.assert_calls()