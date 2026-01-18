import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def create_db_instance(self, db_instance_identifier, allocated_storage, db_instance_class, engine, master_username, master_user_password, db_name=None, db_security_groups=None, vpc_security_group_ids=None, availability_zone=None, db_subnet_group_name=None, preferred_maintenance_window=None, db_parameter_group_name=None, backup_retention_period=None, preferred_backup_window=None, port=None, multi_az=None, engine_version=None, auto_minor_version_upgrade=None, license_model=None, iops=None, option_group_name=None, character_set_name=None, publicly_accessible=None, tags=None):
    """
        Creates a new DB instance.

        :type db_name: string
        :param db_name: The meaning of this parameter differs according to the
            database engine you use.
        **MySQL**

        The name of the database to create when the DB instance is created. If
            this parameter is not specified, no database is created in the DB
            instance.

        Constraints:


        + Must contain 1 to 64 alphanumeric characters
        + Cannot be a word reserved by the specified database engine


        Type: String

        **Oracle**

        The Oracle System ID (SID) of the created DB instance.

        Default: `ORCL`

        Constraints:


        + Cannot be longer than 8 characters


        **SQL Server**

        Not applicable. Must be null.

        :type db_instance_identifier: string
        :param db_instance_identifier: The DB instance identifier. This
            parameter is stored as a lowercase string.
        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens (1 to 15
              for SQL Server).
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.


        Example: `mydbinstance`

        :type allocated_storage: integer
        :param allocated_storage: The amount of storage (in gigabytes) to be
            initially allocated for the database instance.
        **MySQL**

        Constraints: Must be an integer from 5 to 1024.

        Type: Integer

        **Oracle**

        Constraints: Must be an integer from 10 to 1024.

        **SQL Server**

        Constraints: Must be an integer from 200 to 1024 (Standard Edition and
            Enterprise Edition) or from 30 to 1024 (Express Edition and Web
            Edition)

        :type db_instance_class: string
        :param db_instance_class: The compute and memory capacity of the DB
            instance.
        Valid Values: `db.t1.micro | db.m1.small | db.m1.medium | db.m1.large |
            db.m1.xlarge | db.m2.xlarge |db.m2.2xlarge | db.m2.4xlarge`

        :type engine: string
        :param engine: The name of the database engine to be used for this
            instance.
        Valid Values: `MySQL` | `oracle-se1` | `oracle-se` | `oracle-ee` |
            `sqlserver-ee` | `sqlserver-se` | `sqlserver-ex` | `sqlserver-web`

        :type master_username: string
        :param master_username:
        The name of master user for the client DB instance.

        **MySQL**

        Constraints:


        + Must be 1 to 16 alphanumeric characters.
        + First character must be a letter.
        + Cannot be a reserved word for the chosen database engine.


        Type: String

        **Oracle**

        Constraints:


        + Must be 1 to 30 alphanumeric characters.
        + First character must be a letter.
        + Cannot be a reserved word for the chosen database engine.


        **SQL Server**

        Constraints:


        + Must be 1 to 128 alphanumeric characters.
        + First character must be a letter.
        + Cannot be a reserved word for the chosen database engine.

        :type master_user_password: string
        :param master_user_password: The password for the master database user.
            Can be any printable ASCII character except "/", '"', or "@".
        Type: String

        **MySQL**

        Constraints: Must contain from 8 to 41 characters.

        **Oracle**

        Constraints: Must contain from 8 to 30 characters.

        **SQL Server**

        Constraints: Must contain from 8 to 128 characters.

        :type db_security_groups: list
        :param db_security_groups: A list of DB security groups to associate
            with this DB instance.
        Default: The default DB security group for the database engine.

        :type vpc_security_group_ids: list
        :param vpc_security_group_ids: A list of EC2 VPC security groups to
            associate with this DB instance.
        Default: The default EC2 VPC security group for the DB subnet group's
            VPC.

        :type availability_zone: string
        :param availability_zone: The EC2 Availability Zone that the database
            instance will be created in.
        Default: A random, system-chosen Availability Zone in the endpoint's
            region.

        Example: `us-east-1d`

        Constraint: The AvailabilityZone parameter cannot be specified if the
            MultiAZ parameter is set to `True`. The specified Availability Zone
            must be in the same region as the current endpoint.

        :type db_subnet_group_name: string
        :param db_subnet_group_name: A DB subnet group to associate with this
            DB instance.
        If there is no DB subnet group, then it is a non-VPC DB instance.

        :type preferred_maintenance_window: string
        :param preferred_maintenance_window: The weekly time range (in UTC)
            during which system maintenance can occur.
        Format: `ddd:hh24:mi-ddd:hh24:mi`

        Default: A 30-minute window selected at random from an 8-hour block of
            time per region, occurring on a random day of the week. To see the
            time blocks available, see ` Adjusting the Preferred Maintenance
            Window`_ in the Amazon RDS User Guide.

        Valid Days: Mon, Tue, Wed, Thu, Fri, Sat, Sun

        Constraints: Minimum 30-minute window.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of the DB parameter group to associate with this DB instance.
            If this argument is omitted, the default DBParameterGroup for the
            specified engine will be used.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type backup_retention_period: integer
        :param backup_retention_period:
        The number of days for which automated backups are retained. Setting
            this parameter to a positive number enables backups. Setting this
            parameter to 0 disables automated backups.

        Default: 1

        Constraints:


        + Must be a value from 0 to 8
        + Cannot be set to 0 if the DB instance is a master instance with read
              replicas

        :type preferred_backup_window: string
        :param preferred_backup_window: The daily time range during which
            automated backups are created if automated backups are enabled,
            using the `BackupRetentionPeriod` parameter.
        Default: A 30-minute window selected at random from an 8-hour block of
            time per region. See the Amazon RDS User Guide for the time blocks
            for each region from which the default backup windows are assigned.

        Constraints: Must be in the format `hh24:mi-hh24:mi`. Times should be
            Universal Time Coordinated (UTC). Must not conflict with the
            preferred maintenance window. Must be at least 30 minutes.

        :type port: integer
        :param port: The port number on which the database accepts connections.
        **MySQL**

        Default: `3306`

        Valid Values: `1150-65535`

        Type: Integer

        **Oracle**

        Default: `1521`

        Valid Values: `1150-65535`

        **SQL Server**

        Default: `1433`

        Valid Values: `1150-65535` except for `1434` and `3389`.

        :type multi_az: boolean
        :param multi_az: Specifies if the DB instance is a Multi-AZ deployment.
            You cannot set the AvailabilityZone parameter if the MultiAZ
            parameter is set to true.

        :type engine_version: string
        :param engine_version: The version number of the database engine to
            use.
        **MySQL**

        Example: `5.1.42`

        Type: String

        **Oracle**

        Example: `11.2.0.2.v2`

        Type: String

        **SQL Server**

        Example: `10.50.2789.0.v1`

        :type auto_minor_version_upgrade: boolean
        :param auto_minor_version_upgrade: Indicates that minor engine upgrades
            will be applied automatically to the DB instance during the
            maintenance window.
        Default: `True`

        :type license_model: string
        :param license_model: License model information for this DB instance.
        Valid values: `license-included` | `bring-your-own-license` | `general-
            public-license`

        :type iops: integer
        :param iops: The amount of Provisioned IOPS (input/output operations
            per second) to be initially allocated for the DB instance.
        Constraints: Must be an integer greater than 1000.

        :type option_group_name: string
        :param option_group_name: Indicates that the DB instance should be
            associated with the specified option group.
        Permanent options, such as the TDE option for Oracle Advanced Security
            TDE, cannot be removed from an option group, and that option group
            cannot be removed from a DB instance once it is associated with a
            DB instance

        :type character_set_name: string
        :param character_set_name: For supported engines, indicates that the DB
            instance should be associated with the specified CharacterSet.

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
    params = {'DBInstanceIdentifier': db_instance_identifier, 'AllocatedStorage': allocated_storage, 'DBInstanceClass': db_instance_class, 'Engine': engine, 'MasterUsername': master_username, 'MasterUserPassword': master_user_password}
    if db_name is not None:
        params['DBName'] = db_name
    if db_security_groups is not None:
        self.build_list_params(params, db_security_groups, 'DBSecurityGroups.member')
    if vpc_security_group_ids is not None:
        self.build_list_params(params, vpc_security_group_ids, 'VpcSecurityGroupIds.member')
    if availability_zone is not None:
        params['AvailabilityZone'] = availability_zone
    if db_subnet_group_name is not None:
        params['DBSubnetGroupName'] = db_subnet_group_name
    if preferred_maintenance_window is not None:
        params['PreferredMaintenanceWindow'] = preferred_maintenance_window
    if db_parameter_group_name is not None:
        params['DBParameterGroupName'] = db_parameter_group_name
    if backup_retention_period is not None:
        params['BackupRetentionPeriod'] = backup_retention_period
    if preferred_backup_window is not None:
        params['PreferredBackupWindow'] = preferred_backup_window
    if port is not None:
        params['Port'] = port
    if multi_az is not None:
        params['MultiAZ'] = str(multi_az).lower()
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if auto_minor_version_upgrade is not None:
        params['AutoMinorVersionUpgrade'] = str(auto_minor_version_upgrade).lower()
    if license_model is not None:
        params['LicenseModel'] = license_model
    if iops is not None:
        params['Iops'] = iops
    if option_group_name is not None:
        params['OptionGroupName'] = option_group_name
    if character_set_name is not None:
        params['CharacterSetName'] = character_set_name
    if publicly_accessible is not None:
        params['PubliclyAccessible'] = str(publicly_accessible).lower()
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='CreateDBInstance', verb='POST', path='/', params=params)