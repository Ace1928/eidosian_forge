import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_volume_by_id(self):
    vol1 = meta.obj_to_munch(fakes.FakeVolume('01', 'available', 'vol1'))
    self.register_uris([self.get_cinder_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['volumes', '01']), json={'volume': vol1})])
    self._compare_volumes(vol1, self.cloud.get_volume_by_id('01'))
    self.assert_calls()