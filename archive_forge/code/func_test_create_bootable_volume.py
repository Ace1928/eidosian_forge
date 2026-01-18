import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_bootable_volume(self):
    vol1 = meta.obj_to_munch(fakes.FakeVolume('01', 'available', 'vol1'))
    self.register_uris([self.get_cinder_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['volumes']), json={'volume': vol1}, validate=dict(json={'volume': {'size': 50, 'name': 'vol1'}})), dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['volumes', '01', 'action']), validate=dict(json={'os-set_bootable': {'bootable': True}}))])
    self.cloud.create_volume(50, name='vol1', bootable=True)
    self.assert_calls()