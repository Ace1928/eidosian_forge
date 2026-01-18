import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_delete_property(self):
    self.controller.delete(NAMESPACE1, PROPERTY1)
    expect = [('DELETE', '/v2/metadefs/namespaces/%s/properties/%s' % (NAMESPACE1, PROPERTY1), {}, None)]
    self.assertEqual(expect, self.api.calls)