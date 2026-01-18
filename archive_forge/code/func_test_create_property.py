import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_create_property(self):
    properties = {'name': PROPERTYNEW, 'title': 'TITLE', 'type': 'string'}
    obj = self.controller.create(NAMESPACE1, **properties)
    self.assertEqual(PROPERTYNEW, obj.name)