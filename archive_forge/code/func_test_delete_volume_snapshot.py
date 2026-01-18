from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_volume_snapshot(self):
    """
        Test that delete_volume_snapshot without a wait returns True instance
        when the volume snapshot deletes.
        """
    fake_snapshot = fakes.FakeVolumeSnapshot('1234', 'available', 'foo', 'derpysnapshot')
    fake_snapshot_dict = meta.obj_to_munch(fake_snapshot)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', 'detail']), json={'snapshots': [fake_snapshot_dict]}), dict(method='DELETE', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', fake_snapshot_dict['id']]))])
    self.assertTrue(self.cloud.delete_volume_snapshot(name_or_id='1234', wait=False))
    self.assert_calls()