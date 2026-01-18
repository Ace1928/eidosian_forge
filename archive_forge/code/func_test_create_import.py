import time
import uuid
from designateclient.tests import v2
def test_create_import(self):
    zonefile = '$ORIGIN example.com'
    parts = ['zones', 'tasks', 'imports']
    self.stub_url('POST', parts=parts, json=zonefile)
    self.client.zone_imports.create(zonefile)
    self.assertRequestBodyIs(body=zonefile)