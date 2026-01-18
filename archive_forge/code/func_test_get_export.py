import time
import uuid
from designateclient.tests import v2
def test_get_export(self):
    ref = self.new_ref()
    parts = ['zones', 'tasks', 'exports', ref['id']]
    self.stub_url('GET', parts=parts, json=ref)
    self.stub_entity('GET', parts=parts, entity=ref, id=ref['id'])
    response = self.client.zone_exports.get_export_record(ref['id'])
    self.assertEqual(ref, response)