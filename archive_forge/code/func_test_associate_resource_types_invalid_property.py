import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_associate_resource_types_invalid_property(self):
    longer = '1234' * 50
    properties = {'name': RESOURCE_TYPENEW, 'prefix': longer}
    self.assertRaises(TypeError, self.controller.associate, NAMESPACE1, **properties)