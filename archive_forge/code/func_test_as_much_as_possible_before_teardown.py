import unittest
import time
from nose.plugins.attrib import attr
from boto.redshift.layer1 import RedshiftConnection
from boto.redshift.exceptions import ClusterNotFoundFault
from boto.redshift.exceptions import ResizeNotFoundFault
@attr('notdefault')
def test_as_much_as_possible_before_teardown(self):
    with self.assertRaises(ClusterNotFoundFault):
        self.api.describe_clusters('badpipelineid')
    cluster_id = self.create_cluster()
    with self.assertRaises(ResizeNotFoundFault):
        self.api.describe_resize(cluster_id)
    clusters = self.api.describe_clusters()['DescribeClustersResponse']['DescribeClustersResult']['Clusters']
    cluster_ids = [c['ClusterIdentifier'] for c in clusters]
    self.assertIn(cluster_id, cluster_ids)
    response = self.api.describe_clusters(cluster_id)
    self.assertEqual(response['DescribeClustersResponse']['DescribeClustersResult']['Clusters'][0]['ClusterIdentifier'], cluster_id)
    snapshot_id = 'snap-%s' % cluster_id
    response = self.api.create_cluster_snapshot(snapshot_id, cluster_id)
    self.assertEqual(response['CreateClusterSnapshotResponse']['CreateClusterSnapshotResult']['Snapshot']['SnapshotIdentifier'], snapshot_id)
    self.assertEqual(response['CreateClusterSnapshotResponse']['CreateClusterSnapshotResult']['Snapshot']['Status'], 'creating')
    self.addCleanup(self.api.delete_cluster_snapshot, snapshot_id)
    time.sleep(self.wait_time)
    response = self.api.describe_cluster_snapshots(cluster_identifier=cluster_id)
    snap = response['DescribeClusterSnapshotsResponse']['DescribeClusterSnapshotsResult']['Snapshots'][-1]
    self.assertEqual(snap['SnapshotType'], 'manual')
    self.assertEqual(snap['DBName'], self.db_name)