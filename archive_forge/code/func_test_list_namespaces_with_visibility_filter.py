import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_namespaces_with_visibility_filter(self):
    namespaces = self.controller.list(filters={'visibility': 'private'})
    self.assertEqual(1, len(namespaces))
    self.assertEqual(NAMESPACE5, namespaces[0]['namespace'])