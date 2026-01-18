import time
import uuid
from designateclient.tests import v2
def test_delete_request(self):
    transfer = '098bee04-fe30-4a83-8ccd-e0c496755816'
    parts = ['zones', 'tasks', 'transfer_requests', transfer]
    self.stub_url('DELETE', parts=parts)
    self.client.zone_transfers.delete_request(transfer)
    self.assertRequestBodyIs('')