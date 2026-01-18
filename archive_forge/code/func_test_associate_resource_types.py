import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_associate_resource_types(self):
    resource_types = self.controller.associate(NAMESPACE1, name=RESOURCE_TYPENEW)
    self.assertEqual(RESOURCE_TYPENEW, resource_types['name'])