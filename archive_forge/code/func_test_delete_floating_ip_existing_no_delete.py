import copy
import datetime
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack import utils
def test_delete_floating_ip_existing_no_delete(self):
    fip_id = '2f245a7b-796b-4f26-9cf9-9e82d248fda7'
    fake_fip = {'id': fip_id, 'floating_ip_address': '172.99.106.167', 'status': 'ACTIVE'}
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips/{0}'.format(fip_id)]), json={}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips']), json={'floatingips': [fake_fip]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips/{0}'.format(fip_id)]), json={}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips']), json={'floatingips': [fake_fip]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips/{0}'.format(fip_id)]), json={}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips']), json={'floatingips': [fake_fip]})])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_floating_ip, floating_ip_id=fip_id, retry=2)
    self.assert_calls()