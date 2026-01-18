import time
import uuid
from designateclient.tests import v2
def test_list_accepts(self):
    accept_id = '098bee04-fe30-4a83-8ccd-e0c496755816'
    ref = {'id': accept_id, 'status': 'COMPLETE'}
    parts = ['zones', 'tasks', 'transfer_accepts']
    self.stub_url('GET', parts=parts, json={'transfer_accepts': ref})
    self.client.zone_transfers.list_accepts()
    self.assertRequestBodyIs('')