from openstack.shared_file_system.v2 import limit
from openstack.tests.unit import base
def test_make_limits(self):
    limits = limit.Limit(**EXAMPLE)
    self.assertEqual(EXAMPLE['totalShareNetworksUsed'], limits.totalShareNetworksUsed)
    self.assertEqual(EXAMPLE['maxTotalShareGigabytes'], limits.maxTotalShareGigabytes)
    self.assertEqual(EXAMPLE['maxTotalShareNetworks'], limits.maxTotalShareNetworks)
    self.assertEqual(EXAMPLE['totalSharesUsed'], limits.totalSharesUsed)
    self.assertEqual(EXAMPLE['totalShareGigabytesUsed'], limits.totalShareGigabytesUsed)
    self.assertEqual(EXAMPLE['totalShareSnapshotsUsed'], limits.totalShareSnapshotsUsed)
    self.assertEqual(EXAMPLE['maxTotalShares'], limits.maxTotalShares)
    self.assertEqual(EXAMPLE['totalSnapshotGigabytesUsed'], limits.totalSnapshotGigabytesUsed)
    self.assertEqual(EXAMPLE['maxTotalSnapshotGigabytes'], limits.maxTotalSnapshotGigabytes)
    self.assertEqual(EXAMPLE['maxTotalShareSnapshots'], limits.maxTotalShareSnapshots)
    self.assertEqual(EXAMPLE['maxTotalShareReplicas'], limits.maxTotalShareReplicas)
    self.assertEqual(EXAMPLE['maxTotalReplicaGigabytes'], limits.maxTotalReplicaGigabytes)
    self.assertEqual(EXAMPLE['totalShareReplicasUsed'], limits.totalShareReplicasUsed)
    self.assertEqual(EXAMPLE['totalReplicaGigabytesUsed'], limits.totalReplicaGigabytesUsed)