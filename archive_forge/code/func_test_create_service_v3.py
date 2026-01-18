from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_create_service_v3(self):
    service_data = self._get_service_data(name='a service', type='network', description='A test service')
    self.register_uris([dict(method='POST', uri=self.get_mock_url(), status_code=200, json=service_data.json_response_v3, validate=dict(json={'service': service_data.json_request}))])
    service = self.cloud.create_service(name=service_data.service_name, service_type=service_data.service_type, description=service_data.description)
    self.assertThat(service.name, matchers.Equals(service_data.service_name))
    self.assertThat(service.id, matchers.Equals(service_data.service_id))
    self.assertThat(service.description, matchers.Equals(service_data.description))
    self.assertThat(service.type, matchers.Equals(service_data.service_type))
    self.assert_calls()