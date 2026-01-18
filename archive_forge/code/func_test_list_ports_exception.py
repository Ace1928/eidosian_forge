from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_list_ports_exception(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports']), status_code=500)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_ports)