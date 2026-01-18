import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_cluster_snapshot(self, snapshot_identifier, snapshot_cluster_identifier=None):
    """
        Deletes the specified manual snapshot. The snapshot must be in
        the `available` state, with no other users authorized to
        access the snapshot.

        Unlike automated snapshots, manual snapshots are retained even
        after you delete your cluster. Amazon Redshift does not delete
        your manual snapshots. You must delete manual snapshot
        explicitly to avoid getting charged. If other accounts are
        authorized to access the snapshot, you must revoke all of the
        authorizations before you can delete the snapshot.

        :type snapshot_identifier: string
        :param snapshot_identifier: The unique identifier of the manual
            snapshot to be deleted.
        Constraints: Must be the name of an existing snapshot that is in the
            `available` state.

        :type snapshot_cluster_identifier: string
        :param snapshot_cluster_identifier: The unique identifier of the
            cluster the snapshot was created from. This parameter is required
            if your IAM user has a policy containing a snapshot resource
            element that specifies anything other than * for the cluster name.
        Constraints: Must be the name of valid cluster.

        """
    params = {'SnapshotIdentifier': snapshot_identifier}
    if snapshot_cluster_identifier is not None:
        params['SnapshotClusterIdentifier'] = snapshot_cluster_identifier
    return self._make_request(action='DeleteClusterSnapshot', verb='POST', path='/', params=params)