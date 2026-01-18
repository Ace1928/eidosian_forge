import copy
import uuid
from openstack.tests.unit import base
def test_bind_accelerator_request(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json={'accelerator_requests': [ARQ_DICT]}), dict(method='PATCH', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json=ARQ_DICT)])
    properties = [{'path': '/hostname', 'value': ARQ_DICT['hostname'], 'op': 'add'}, {'path': '/instance_uuid', 'value': ARQ_DICT['instance_uuid'], 'op': 'add'}, {'path': '/device_rp_uuid', 'value': ARQ_DICT['device_rp_uuid'], 'op': 'add'}]
    self.assertTrue(self.cloud.bind_accelerator_request(ARQ_DICT['uuid'], properties))
    self.assert_calls()