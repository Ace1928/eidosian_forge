import time
import uuid
from designateclient.tests import v2
def test_create_request_with_description(self):
    zone = '098bee04-fe30-4a83-8ccd-e0c496755816'
    project = '123'
    ref = {'target_project_id': project, 'description': 'My Foo'}
    parts = ['zones', zone, 'tasks', 'transfer_requests']
    self.stub_url('POST', parts=parts, json=ref)
    self.client.zone_transfers.create_request(zone, project, ref['description'])
    self.assertRequestBodyIs(json=ref)