from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_endpoint_set(self):
    endpoint_id = self._create_dummy_endpoint()
    new_endpoint_url = data_utils.rand_url()
    raw_output = self.openstack('endpoint set --interface %(interface)s --url %(url)s --disable %(endpoint_id)s' % {'interface': 'admin', 'url': new_endpoint_url, 'endpoint_id': endpoint_id})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('endpoint show %s' % endpoint_id)
    endpoint = self.parse_show_as_object(raw_output)
    self.assertEqual('admin', endpoint['interface'])
    self.assertEqual(new_endpoint_url, endpoint['url'])
    self.assertEqual('False', endpoint['enabled'])