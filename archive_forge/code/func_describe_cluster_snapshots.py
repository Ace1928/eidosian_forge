import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_cluster_snapshots(self, cluster_identifier=None, snapshot_identifier=None, snapshot_type=None, start_time=None, end_time=None, max_records=None, marker=None, owner_account=None):
    """
        Returns one or more snapshot objects, which contain metadata
        about your cluster snapshots. By default, this operation
        returns information about all snapshots of all clusters that
        are owned by you AWS customer account. No information is
        returned for snapshots owned by inactive AWS customer
        accounts.

        :type cluster_identifier: string
        :param cluster_identifier: The identifier of the cluster for which
            information about snapshots is requested.

        :type snapshot_identifier: string
        :param snapshot_identifier: The snapshot identifier of the snapshot
            about which to return information.

        :type snapshot_type: string
        :param snapshot_type: The type of snapshots for which you are
            requesting information. By default, snapshots of all types are
            returned.
        Valid Values: `automated` | `manual`

        :type start_time: timestamp
        :param start_time: A value that requests only snapshots created at or
            after the specified time. The time value is specified in ISO 8601
            format. For more information about ISO 8601, go to the `ISO8601
            Wikipedia page.`_
        Example: `2012-07-16T18:00:00Z`

        :type end_time: timestamp
        :param end_time: A time value that requests only snapshots created at
            or before the specified time. The time value is specified in ISO
            8601 format. For more information about ISO 8601, go to the
            `ISO8601 Wikipedia page.`_
        Example: `2012-07-16T18:00:00Z`

        :type max_records: integer
        :param max_records: The maximum number of response records to return in
            each call. If the number of remaining response records exceeds the
            specified `MaxRecords` value, a value is returned in a `marker`
            field of the response. You can retrieve the next set of records by
            retrying the command with the returned marker value.
        Default: `100`

        Constraints: minimum 20, maximum 100.

        :type marker: string
        :param marker: An optional parameter that specifies the starting point
            to return a set of response records. When the results of a
            DescribeClusterSnapshots request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        :type owner_account: string
        :param owner_account: The AWS customer account used to create or copy
            the snapshot. Use this field to filter the results to snapshots
            owned by a particular account. To describe snapshots you own,
            either specify your AWS customer account, or do not specify the
            parameter.

        """
    params = {}
    if cluster_identifier is not None:
        params['ClusterIdentifier'] = cluster_identifier
    if snapshot_identifier is not None:
        params['SnapshotIdentifier'] = snapshot_identifier
    if snapshot_type is not None:
        params['SnapshotType'] = snapshot_type
    if start_time is not None:
        params['StartTime'] = start_time
    if end_time is not None:
        params['EndTime'] = end_time
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    if owner_account is not None:
        params['OwnerAccount'] = owner_account
    return self._make_request(action='DescribeClusterSnapshots', verb='POST', path='/', params=params)