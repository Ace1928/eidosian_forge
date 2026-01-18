import time
import uuid
from designateclient.tests import v2
def test_accept_request(self):
    transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
    key = 'foo123'
    ref = {'status': 'COMPLETE'}
    parts = ['zones', 'tasks', 'transfer_accepts']
    self.stub_url('POST', parts=parts, json=ref)
    request = {'key': key, 'zone_transfer_request_id': transfer}
    self.client.zone_transfers.accept_request(transfer, key)
    self.assertRequestBodyIs(json=request)