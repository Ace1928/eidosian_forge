import time
import uuid
from designateclient.tests import v2
def test_get_import(self):
    ref = self.new_ref()
    parts = ['zones', 'tasks', 'imports', ref['id']]
    self.stub_url('GET', parts=parts, json=ref)
    self.stub_entity('GET', parts=parts, entity=ref, id=ref['id'])
    response = self.client.zone_imports.get_import_record(ref['id'])
    self.assertEqual(ref, response)