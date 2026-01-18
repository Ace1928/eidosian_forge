import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def create_cache_cluster(self, cache_cluster_id, num_cache_nodes=None, cache_node_type=None, engine=None, replication_group_id=None, engine_version=None, cache_parameter_group_name=None, cache_subnet_group_name=None, cache_security_group_names=None, security_group_ids=None, snapshot_arns=None, preferred_availability_zone=None, preferred_maintenance_window=None, port=None, notification_topic_arn=None, auto_minor_version_upgrade=None):
    """
        The CreateCacheCluster operation creates a new cache cluster.
        All nodes in the cache cluster run the same protocol-compliant
        cache engine software - either Memcached or Redis.

        :type cache_cluster_id: string
        :param cache_cluster_id:
        The cache cluster identifier. This parameter is stored as a lowercase
            string.

        Constraints:


        + Must contain from 1 to 20 alphanumeric characters or hyphens.
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.

        :type replication_group_id: string
        :param replication_group_id: The replication group to which this cache
            cluster should belong. If this parameter is specified, the cache
            cluster will be added to the specified replication group as a read
            replica; otherwise, the cache cluster will be a standalone primary
            that is not part of any replication group.

        :type num_cache_nodes: integer
        :param num_cache_nodes: The initial number of cache nodes that the
            cache cluster will have.
        For a Memcached cluster, valid values are between 1 and 20. If you need
            to exceed this limit, please fill out the ElastiCache Limit
            Increase Request form at ``_ .

        For Redis, only single-node cache clusters are supported at this time,
            so the value for this parameter must be 1.

        :type cache_node_type: string
        :param cache_node_type: The compute and memory capacity of the nodes in
            the cache cluster.
        Valid values for Memcached:

        `cache.t1.micro` | `cache.m1.small` | `cache.m1.medium` |
            `cache.m1.large` | `cache.m1.xlarge` | `cache.m3.xlarge` |
            `cache.m3.2xlarge` | `cache.m2.xlarge` | `cache.m2.2xlarge` |
            `cache.m2.4xlarge` | `cache.c1.xlarge`

        Valid values for Redis:

        `cache.t1.micro` | `cache.m1.small` | `cache.m1.medium` |
            `cache.m1.large` | `cache.m1.xlarge` | `cache.m2.xlarge` |
            `cache.m2.2xlarge` | `cache.m2.4xlarge` | `cache.c1.xlarge`

        For a complete listing of cache node types and specifications, see `.

        :type engine: string
        :param engine: The name of the cache engine to be used for this cache
            cluster.
        Valid values for this parameter are:

        `memcached` | `redis`

        :type engine_version: string
        :param engine_version: The version number of the cache engine to be
            used for this cluster. To view the supported cache engine versions,
            use the DescribeCacheEngineVersions operation.

        :type cache_parameter_group_name: string
        :param cache_parameter_group_name: The name of the cache parameter
            group to associate with this cache cluster. If this argument is
            omitted, the default cache parameter group for the specified engine
            will be used.

        :type cache_subnet_group_name: string
        :param cache_subnet_group_name: The name of the cache subnet group to
            be used for the cache cluster.
        Use this parameter only when you are creating a cluster in an Amazon
            Virtual Private Cloud (VPC).

        :type cache_security_group_names: list
        :param cache_security_group_names: A list of cache security group names
            to associate with this cache cluster.
        Use this parameter only when you are creating a cluster outside of an
            Amazon Virtual Private Cloud (VPC).

        :type security_group_ids: list
        :param security_group_ids: One or more VPC security groups associated
            with the cache cluster.
        Use this parameter only when you are creating a cluster in an Amazon
            Virtual Private Cloud (VPC).

        :type snapshot_arns: list
        :param snapshot_arns: A single-element string list containing an Amazon
            Resource Name (ARN) that uniquely identifies a Redis RDB snapshot
            file stored in Amazon S3. The snapshot file will be used to
            populate the Redis cache in the new cache cluster. The Amazon S3
            object name in the ARN cannot contain any commas.
        Here is an example of an Amazon S3 ARN:
            `arn:aws:s3:::my_bucket/snapshot1.rdb`

        **Note:** This parameter is only valid if the `Engine` parameter is
            `redis`.

        :type preferred_availability_zone: string
        :param preferred_availability_zone: The EC2 Availability Zone in which
            the cache cluster will be created.
        All cache nodes belonging to a cache cluster are placed in the
            preferred availability zone.

        Default: System chosen availability zone.

        :type preferred_maintenance_window: string
        :param preferred_maintenance_window: The weekly time range (in UTC)
            during which system maintenance can occur.
        Example: `sun:05:00-sun:09:00`

        :type port: integer
        :param port: The port number on which each of the cache nodes will
            accept connections.

        :type notification_topic_arn: string
        :param notification_topic_arn:
        The Amazon Resource Name (ARN) of the Amazon Simple Notification
            Service (SNS) topic to which notifications will be sent.

        The Amazon SNS topic owner must be the same as the cache cluster owner.

        :type auto_minor_version_upgrade: boolean
        :param auto_minor_version_upgrade: Determines whether minor engine
            upgrades will be applied automatically to the cache cluster during
            the maintenance window. A value of `True` allows these upgrades to
            occur; `False` disables automatic upgrades.
        Default: `True`

        """
    params = {'CacheClusterId': cache_cluster_id}
    if num_cache_nodes is not None:
        params['NumCacheNodes'] = num_cache_nodes
    if cache_node_type is not None:
        params['CacheNodeType'] = cache_node_type
    if engine is not None:
        params['Engine'] = engine
    if replication_group_id is not None:
        params['ReplicationGroupId'] = replication_group_id
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if cache_parameter_group_name is not None:
        params['CacheParameterGroupName'] = cache_parameter_group_name
    if cache_subnet_group_name is not None:
        params['CacheSubnetGroupName'] = cache_subnet_group_name
    if cache_security_group_names is not None:
        self.build_list_params(params, cache_security_group_names, 'CacheSecurityGroupNames.member')
    if security_group_ids is not None:
        self.build_list_params(params, security_group_ids, 'SecurityGroupIds.member')
    if snapshot_arns is not None:
        self.build_list_params(params, snapshot_arns, 'SnapshotArns.member')
    if preferred_availability_zone is not None:
        params['PreferredAvailabilityZone'] = preferred_availability_zone
    if preferred_maintenance_window is not None:
        params['PreferredMaintenanceWindow'] = preferred_maintenance_window
    if port is not None:
        params['Port'] = port
    if notification_topic_arn is not None:
        params['NotificationTopicArn'] = notification_topic_arn
    if auto_minor_version_upgrade is not None:
        params['AutoMinorVersionUpgrade'] = str(auto_minor_version_upgrade).lower()
    return self._make_request(action='CreateCacheCluster', verb='POST', path='/', params=params)