import time
import uuid
from designateclient.tests import v2
def test_list_imports(self):
    items = [self.new_ref(), self.new_ref()]
    parts = ['zones', 'tasks', 'imports']
    self.stub_url('GET', parts=parts, json={'imports': items})
    listed = self.client.zone_imports.list()
    self.assertList(items, listed['imports'])
    self.assertQueryStringIs('')