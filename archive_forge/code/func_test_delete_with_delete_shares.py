import time
import uuid
from designateclient.tests import v2
def test_delete_with_delete_shares(self):
    ref = self.new_ref()
    self.stub_entity('DELETE', id=ref['id'])
    self.client.zones.delete(ref['id'], delete_shares=True)
    self.assertRequestBodyIs(None)
    self.assertRequestHeaderEqual('X-Designate-Delete-Shares', 'true')