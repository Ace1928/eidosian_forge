import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_domain_name_or_id(self):
    domain_data = self._get_domain_data()
    new_resp = domain_data.json_response.copy()
    new_resp['domain']['enabled'] = False
    domain_resource_uri = self.get_mock_url(append=[domain_data.domain_id])
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[domain_data.domain_id]), status_code=200, json={'domain': domain_data.json_response['domain']}), dict(method='PATCH', uri=domain_resource_uri, status_code=200, json=new_resp, validate=dict(json={'domain': {'enabled': False}})), dict(method='DELETE', uri=domain_resource_uri, status_code=204)])
    self.cloud.delete_domain(name_or_id=domain_data.domain_id)
    self.assert_calls()