import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_namespaces_with_multiple_resource_types_filter(self):
    namespaces = self.controller.list(filters={'resource_types': [RESOURCE_TYPE1, RESOURCE_TYPE2]})
    self.assertEqual(1, len(namespaces))
    self.assertEqual(NAMESPACE4, namespaces[0]['namespace'])