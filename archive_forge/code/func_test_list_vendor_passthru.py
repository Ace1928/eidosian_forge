from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import driver
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
def test_list_vendor_passthru(self):
    self.session = mock.Mock(spec=adapter.Adapter)
    sot = driver.Driver(**FAKE)
    fake_vendor_passthru_info = {'fake_vendor_method': {'async': True, 'attach': False, 'description': 'Fake function that does nothing in background', 'http_methods': ['GET', 'PUT', 'POST', 'DELETE']}}
    self.session.get.return_value.json.return_value = fake_vendor_passthru_info
    result = sot.list_vendor_passthru(self.session)
    self.session.get.assert_called_once_with('drivers/{driver_name}/vendor_passthru/methods'.format(driver_name=FAKE['name']), headers=mock.ANY)
    self.assertEqual(result, fake_vendor_passthru_info)