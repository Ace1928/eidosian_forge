import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import services
def test_create_without_description(self):
    req_body = {'OS-KSADM:service': {'name': 'swift', 'type': 'object-store', 'description': None}}
    service_id = uuid.uuid4().hex
    resp_body = {'OS-KSADM:service': {'name': 'swift', 'type': 'object-store', 'id': service_id, 'description': None}}
    self.stub_url('POST', ['OS-KSADM', 'services'], json=resp_body)
    service = self.client.services.create(req_body['OS-KSADM:service']['name'], req_body['OS-KSADM:service']['type'], req_body['OS-KSADM:service']['description'])
    self.assertIsInstance(service, services.Service)
    self.assertEqual(service.id, service_id)
    self.assertEqual(service.name, req_body['OS-KSADM:service']['name'])
    self.assertIsNone(service.description)
    self.assertRequestBodyIs(json=req_body)