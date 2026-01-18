import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def delete_replication_group(self, replication_group_id):
    """
        The DeleteReplicationGroup operation deletes an existing
        replication group. DeleteReplicationGroup deletes the primary
        cache cluster and all of the read replicas in the replication
        group. When you receive a successful response from this
        operation, Amazon ElastiCache immediately begins deleting the
        entire replication group; you cannot cancel or revert this
        operation.

        :type replication_group_id: string
        :param replication_group_id: The identifier for the replication group
            to be deleted. This parameter is not case sensitive.

        """
    params = {'ReplicationGroupId': replication_group_id}
    return self._make_request(action='DeleteReplicationGroup', verb='POST', path='/', params=params)