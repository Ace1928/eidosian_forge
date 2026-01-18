import time
import uuid
from designateclient.tests import v2
def test_delete_import(self):
    ref = self.new_ref()
    parts = ['zones', 'tasks', 'imports', ref['id']]
    self.stub_url('DELETE', parts=parts, json=ref)
    self.stub_entity('DELETE', parts=parts, id=ref['id'])
    self.client.zone_imports.delete(ref['id'])
    self.assertRequestBodyIs(None)