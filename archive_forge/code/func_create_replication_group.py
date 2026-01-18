import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def create_replication_group(self, replication_group_id, primary_cluster_id, replication_group_description):
    """
        The CreateReplicationGroup operation creates a replication
        group. A replication group is a collection of cache clusters,
        where one of the clusters is a read/write primary and the
        other clusters are read-only replicas. Writes to the primary
        are automatically propagated to the replicas.

        When you create a replication group, you must specify an
        existing cache cluster that is in the primary role. When the
        replication group has been successfully created, you can add
        one or more read replica replicas to it, up to a total of five
        read replicas.

        :type replication_group_id: string
        :param replication_group_id:
        The replication group identifier. This parameter is stored as a
            lowercase string.

        Constraints:


        + Must contain from 1 to 20 alphanumeric characters or hyphens.
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.

        :type primary_cluster_id: string
        :param primary_cluster_id: The identifier of the cache cluster that
            will serve as the primary for this replication group. This cache
            cluster must already exist and have a status of available .

        :type replication_group_description: string
        :param replication_group_description: A user-specified description for
            the replication group.

        """
    params = {'ReplicationGroupId': replication_group_id, 'PrimaryClusterId': primary_cluster_id, 'ReplicationGroupDescription': replication_group_description}
    return self._make_request(action='CreateReplicationGroup', verb='POST', path='/', params=params)