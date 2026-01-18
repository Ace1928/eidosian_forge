import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_with_sort_key(self):
    namespaces = self.controller.list(sort_key='created_at', limit=1)
    self.assertEqual(1, len(namespaces))
    self.assertEqual(NAMESPACE1, namespaces[0]['namespace'])