import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_delete_all_objects(self):
    self.controller.delete_all(NAMESPACE1)
    expect = [('DELETE', '/v2/metadefs/namespaces/%s/objects' % NAMESPACE1, {}, None)]
    self.assertEqual(expect, self.api.calls)