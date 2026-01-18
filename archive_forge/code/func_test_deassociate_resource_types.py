import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_deassociate_resource_types(self):
    self.controller.deassociate(NAMESPACE1, RESOURCE_TYPE1)
    expect = [('DELETE', '/v2/metadefs/namespaces/%s/resource_types/%s' % (NAMESPACE1, RESOURCE_TYPE1), {}, None)]
    self.assertEqual(expect, self.api.calls)