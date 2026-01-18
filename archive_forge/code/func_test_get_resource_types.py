import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_get_resource_types(self):
    resource_types = self.controller.get(NAMESPACE1)
    self.assertEqual([RESOURCE_TYPE3, RESOURCE_TYPE4], resource_types)