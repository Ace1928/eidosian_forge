from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_get_service(self):
    service_data = self._get_service_data()
    service2_data = self._get_service_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service'], service2_data.json_response_v3['service']]}), dict(method='GET', uri=self.get_mock_url(), status_code=400)])
    service = self.cloud.get_service(name_or_id=service_data.service_id)
    self.assertThat(service.id, matchers.Equals(service_data.service_id))
    service = self.cloud.get_service(name_or_id=service_data.service_name)
    self.assertThat(service.id, matchers.Equals(service_data.service_id))
    service = self.cloud.get_service(name_or_id='INVALID SERVICE')
    self.assertIs(None, service)
    self.assertRaises(exceptions.SDKException, self.cloud.get_service, name_or_id=None, filters={'type': 'type2'})
    self.assert_calls()