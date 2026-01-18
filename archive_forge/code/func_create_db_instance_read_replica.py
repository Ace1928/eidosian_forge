import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def create_db_instance_read_replica(self, db_instance_identifier, source_db_instance_identifier, db_instance_class=None, availability_zone=None, port=None, auto_minor_version_upgrade=None, iops=None, option_group_name=None, publicly_accessible=None, tags=None):
    """
        Creates a DB instance that acts as a read replica of a source
        DB instance.

        All read replica DB instances are created as Single-AZ
        deployments with backups disabled. All other DB instance
        attributes (including DB security groups and DB parameter
        groups) are inherited from the source DB instance, except as
        specified below.

        The source DB instance must have backup retention enabled.

        :type db_instance_identifier: string
        :param db_instance_identifier: The DB instance identifier of the read
            replica. This is the unique key that identifies a DB instance. This
            parameter is stored as a lowercase string.

        :type source_db_instance_identifier: string
        :param source_db_instance_identifier: The identifier of the DB instance
            that will act as the source for the read replica. Each DB instance
            can have up to five read replicas.
        Constraints: Must be the identifier of an existing DB instance that is
            not already a read replica DB instance.

        :type db_instance_class: string
        :param db_instance_class: The compute and memory capacity of the read
            replica.
        Valid Values: `db.m1.small | db.m1.medium | db.m1.large | db.m1.xlarge
            | db.m2.xlarge |db.m2.2xlarge | db.m2.4xlarge`

        Default: Inherits from the source DB instance.

        :type availability_zone: string
        :param availability_zone: The Amazon EC2 Availability Zone that the
            read replica will be created in.
        Default: A random, system-chosen Availability Zone in the endpoint's
            region.

        Example: `us-east-1d`

        :type port: integer
        :param port: The port number that the DB instance uses for connections.
        Default: Inherits from the source DB instance

        Valid Values: `1150-65535`

        :type auto_minor_version_upgrade: boolean
        :param auto_minor_version_upgrade: Indicates that minor engine upgrades
            will be applied automatically to the read replica during the
            maintenance window.
        Default: Inherits from the source DB instance

        :type iops: integer
        :param iops: The amount of Provisioned IOPS (input/output operations
            per second) to be initially allocated for the DB instance.

        :type option_group_name: string
        :param option_group_name: The option group the DB instance will be
            associated with. If omitted, the default option group for the
            engine specified will be used.

        :type publicly_accessible: boolean
        :param publicly_accessible: Specifies the accessibility options for the
            DB instance. A value of true specifies an Internet-facing instance
            with a publicly resolvable DNS name, which resolves to a public IP
            address. A value of false specifies an internal instance with a DNS
            name that resolves to a private IP address.
        Default: The default behavior varies depending on whether a VPC has
            been requested or not. The following list shows the default
            behavior in each case.


        + **Default VPC:**true
        + **VPC:**false


        If no DB subnet group has been specified as part of the request and the
            PubliclyAccessible value has not been set, the DB instance will be
            publicly accessible. If a specific DB subnet group has been
            specified as part of the request and the PubliclyAccessible value
            has not been set, the DB instance will be private.

        :type tags: list
        :param tags: A list of tags. Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]

        """
    params = {'DBInstanceIdentifier': db_instance_identifier, 'SourceDBInstanceIdentifier': source_db_instance_identifier}
    if db_instance_class is not None:
        params['DBInstanceClass'] = db_instance_class
    if availability_zone is not None:
        params['AvailabilityZone'] = availability_zone
    if port is not None:
        params['Port'] = port
    if auto_minor_version_upgrade is not None:
        params['AutoMinorVersionUpgrade'] = str(auto_minor_version_upgrade).lower()
    if iops is not None:
        params['Iops'] = iops
    if option_group_name is not None:
        params['OptionGroupName'] = option_group_name
    if publicly_accessible is not None:
        params['PubliclyAccessible'] = str(publicly_accessible).lower()
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='CreateDBInstanceReadReplica', verb='POST', path='/', params=params)