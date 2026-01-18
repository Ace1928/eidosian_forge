import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_create_property_invalid_property(self):
    properties = {'namespace': NAMESPACE1}
    self.assertRaises(TypeError, self.controller.create, **properties)