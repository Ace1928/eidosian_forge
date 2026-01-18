from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_has_volume(self):
    mock_cloud = mock.MagicMock()
    fake_volume = fakes.FakeVolume(id='volume1', status='available', name='Volume 1 Display Name', attachments=[{'device': '/dev/sda0'}])
    fake_volume_dict = meta.obj_to_munch(fake_volume)
    mock_cloud.get_volumes.return_value = [fake_volume_dict]
    hostvars = meta.get_hostvars_from_server(mock_cloud, meta.obj_to_munch(standard_fake_server))
    self.assertEqual('volume1', hostvars['volumes'][0]['id'])
    self.assertEqual('/dev/sda0', hostvars['volumes'][0]['device'])