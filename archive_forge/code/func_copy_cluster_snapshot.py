import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def copy_cluster_snapshot(self, source_snapshot_identifier, target_snapshot_identifier, source_snapshot_cluster_identifier=None):
    """
        Copies the specified automated cluster snapshot to a new
        manual cluster snapshot. The source must be an automated
        snapshot and it must be in the available state.

        When you delete a cluster, Amazon Redshift deletes any
        automated snapshots of the cluster. Also, when the retention
        period of the snapshot expires, Amazon Redshift automatically
        deletes it. If you want to keep an automated snapshot for a
        longer period, you can make a manual copy of the snapshot.
        Manual snapshots are retained until you delete them.

        For more information about working with snapshots, go to
        `Amazon Redshift Snapshots`_ in the Amazon Redshift Management
        Guide .

        :type source_snapshot_identifier: string
        :param source_snapshot_identifier:
        The identifier for the source snapshot.

        Constraints:


        + Must be the identifier for a valid automated snapshot whose state is
              `available`.

        :type source_snapshot_cluster_identifier: string
        :param source_snapshot_cluster_identifier:
        The identifier of the cluster the source snapshot was created from.
            This parameter is required if your IAM user has a policy containing
            a snapshot resource element that specifies anything other than *
            for the cluster name.

        Constraints:


        + Must be the identifier for a valid cluster.

        :type target_snapshot_identifier: string
        :param target_snapshot_identifier:
        The identifier given to the new manual snapshot.

        Constraints:


        + Cannot be null, empty, or blank.
        + Must contain from 1 to 255 alphanumeric characters or hyphens.
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.
        + Must be unique for the AWS account that is making the request.

        """
    params = {'SourceSnapshotIdentifier': source_snapshot_identifier, 'TargetSnapshotIdentifier': target_snapshot_identifier}
    if source_snapshot_cluster_identifier is not None:
        params['SourceSnapshotClusterIdentifier'] = source_snapshot_cluster_identifier
    return self._make_request(action='CopyClusterSnapshot', verb='POST', path='/', params=params)