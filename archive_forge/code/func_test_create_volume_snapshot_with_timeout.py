from openstack.block_storage.v3 import snapshot
from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_volume_snapshot_with_timeout(self):
    """
        Test that a timeout while waiting for the volume snapshot to create
        raises an exception in create_volume_snapshot.
        """
    snapshot_id = '5678'
    volume_id = '1234'
    build_snapshot = fakes.FakeVolumeSnapshot(snapshot_id, 'creating', 'foo', 'derpysnapshot')
    build_snapshot_dict = meta.obj_to_munch(build_snapshot)
    self.register_uris([dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['snapshots']), json={'snapshot': build_snapshot_dict}, validate=dict(json={'snapshot': {'volume_id': '1234', 'force': False}})), dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['snapshots', snapshot_id]), json={'snapshot': build_snapshot_dict})])
    self.assertRaises(exceptions.ResourceTimeout, self.cloud.create_volume_snapshot, volume_id=volume_id, wait=True, timeout=0.01)
    self.assert_calls(do_count=False)