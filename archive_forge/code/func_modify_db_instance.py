import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def modify_db_instance(self, db_instance_identifier, allocated_storage=None, db_instance_class=None, db_security_groups=None, vpc_security_group_ids=None, apply_immediately=None, master_user_password=None, db_parameter_group_name=None, backup_retention_period=None, preferred_backup_window=None, preferred_maintenance_window=None, multi_az=None, engine_version=None, allow_major_version_upgrade=None, auto_minor_version_upgrade=None, iops=None, option_group_name=None, new_db_instance_identifier=None):
    """
        Modify settings for a DB instance. You can change one or more
        database configuration parameters by specifying these
        parameters and the new values in the request.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The DB instance identifier. This value is stored as a lowercase string.

        Constraints:


        + Must be the identifier for an existing DB instance
        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type allocated_storage: integer
        :param allocated_storage: The new storage capacity of the RDS instance.
            Changing this parameter does not result in an outage and the change
            is applied during the next maintenance window unless the
            `ApplyImmediately` parameter is set to `True` for this request.
        **MySQL**

        Default: Uses existing setting

        Valid Values: 5-1024

        Constraints: Value supplied must be at least 10% greater than the
            current value. Values that are not at least 10% greater than the
            existing value are rounded up so that they are 10% greater than the
            current value.

        Type: Integer

        **Oracle**

        Default: Uses existing setting

        Valid Values: 10-1024

        Constraints: Value supplied must be at least 10% greater than the
            current value. Values that are not at least 10% greater than the
            existing value are rounded up so that they are 10% greater than the
            current value.

        **SQL Server**

        Cannot be modified.

        If you choose to migrate your DB instance from using standard storage
            to using Provisioned IOPS, or from using Provisioned IOPS to using
            standard storage, the process can take time. The duration of the
            migration depends on several factors such as database load, storage
            size, storage type (standard or Provisioned IOPS), amount of IOPS
            provisioned (if any), and the number of prior scale storage
            operations. Typical migration times are under 24 hours, but the
            process can take up to several days in some cases. During the
            migration, the DB instance will be available for use, but may
            experience performance degradation. While the migration takes
            place, nightly backups for the instance will be suspended. No other
            Amazon RDS operations can take place for the instance, including
            modifying the instance, rebooting the instance, deleting the
            instance, creating a read replica for the instance, and creating a
            DB snapshot of the instance.

        :type db_instance_class: string
        :param db_instance_class: The new compute and memory capacity of the DB
            instance. To determine the instance classes that are available for
            a particular DB engine, use the DescribeOrderableDBInstanceOptions
            action.
        Passing a value for this parameter causes an outage during the change
            and is applied during the next maintenance window, unless the
            `ApplyImmediately` parameter is specified as `True` for this
            request.

        Default: Uses existing setting

        Valid Values: `db.t1.micro | db.m1.small | db.m1.medium | db.m1.large |
            db.m1.xlarge | db.m2.xlarge | db.m2.2xlarge | db.m2.4xlarge`

        :type db_security_groups: list
        :param db_security_groups:
        A list of DB security groups to authorize on this DB instance. Changing
            this parameter does not result in an outage and the change is
            asynchronously applied as soon as possible.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type vpc_security_group_ids: list
        :param vpc_security_group_ids:
        A list of EC2 VPC security groups to authorize on this DB instance.
            This change is asynchronously applied as soon as possible.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type apply_immediately: boolean
        :param apply_immediately: Specifies whether or not the modifications in
            this request and any pending modifications are asynchronously
            applied as soon as possible, regardless of the
            `PreferredMaintenanceWindow` setting for the DB instance.
        If this parameter is passed as `False`, changes to the DB instance are
            applied on the next call to RebootDBInstance, the next maintenance
            reboot, or the next failure reboot, whichever occurs first. See
            each parameter to determine when a change is applied.

        Default: `False`

        :type master_user_password: string
        :param master_user_password:
        The new password for the DB instance master user. Can be any printable
            ASCII character except "/", '"', or "@".

        Changing this parameter does not result in an outage and the change is
            asynchronously applied as soon as possible. Between the time of the
            request and the completion of the request, the `MasterUserPassword`
            element exists in the `PendingModifiedValues` element of the
            operation response.

        Default: Uses existing setting

        Constraints: Must be 8 to 41 alphanumeric characters (MySQL), 8 to 30
            alphanumeric characters (Oracle), or 8 to 128 alphanumeric
            characters (SQL Server).

        Amazon RDS API actions never return the password, so this action
            provides a way to regain access to a master instance user if the
            password is lost.

        :type db_parameter_group_name: string
        :param db_parameter_group_name: The name of the DB parameter group to
            apply to this DB instance. Changing this parameter does not result
            in an outage and the change is applied during the next maintenance
            window unless the `ApplyImmediately` parameter is set to `True` for
            this request.
        Default: Uses existing setting

        Constraints: The DB parameter group must be in the same DB parameter
            group family as this DB instance.

        :type backup_retention_period: integer
        :param backup_retention_period:
        The number of days to retain automated backups. Setting this parameter
            to a positive number enables backups. Setting this parameter to 0
            disables automated backups.

        Changing this parameter can result in an outage if you change from 0 to
            a non-zero value or from a non-zero value to 0. These changes are
            applied during the next maintenance window unless the
            `ApplyImmediately` parameter is set to `True` for this request. If
            you change the parameter from one non-zero value to another non-
            zero value, the change is asynchronously applied as soon as
            possible.

        Default: Uses existing setting

        Constraints:


        + Must be a value from 0 to 8
        + Cannot be set to 0 if the DB instance is a master instance with read
              replicas or if the DB instance is a read replica

        :type preferred_backup_window: string
        :param preferred_backup_window:
        The daily time range during which automated backups are created if
            automated backups are enabled, as determined by the
            `BackupRetentionPeriod`. Changing this parameter does not result in
            an outage and the change is asynchronously applied as soon as
            possible.

        Constraints:


        + Must be in the format hh24:mi-hh24:mi
        + Times should be Universal Time Coordinated (UTC)
        + Must not conflict with the preferred maintenance window
        + Must be at least 30 minutes

        :type preferred_maintenance_window: string
        :param preferred_maintenance_window: The weekly time range (in UTC)
            during which system maintenance can occur, which may result in an
            outage. Changing this parameter does not result in an outage,
            except in the following situation, and the change is asynchronously
            applied as soon as possible. If there are pending actions that
            cause a reboot, and the maintenance window is changed to include
            the current time, then changing this parameter will cause a reboot
            of the DB instance. If moving this window to the current time,
            there must be at least 30 minutes between the current time and end
            of the window to ensure pending changes are applied.
        Default: Uses existing setting

        Format: ddd:hh24:mi-ddd:hh24:mi

        Valid Days: Mon | Tue | Wed | Thu | Fri | Sat | Sun

        Constraints: Must be at least 30 minutes

        :type multi_az: boolean
        :param multi_az: Specifies if the DB instance is a Multi-AZ deployment.
            Changing this parameter does not result in an outage and the change
            is applied during the next maintenance window unless the
            `ApplyImmediately` parameter is set to `True` for this request.
        Constraints: Cannot be specified if the DB instance is a read replica.

        :type engine_version: string
        :param engine_version: The version number of the database engine to
            upgrade to. Changing this parameter results in an outage and the
            change is applied during the next maintenance window unless the
            `ApplyImmediately` parameter is set to `True` for this request.
        For major version upgrades, if a non-default DB parameter group is
            currently in use, a new DB parameter group in the DB parameter
            group family for the new engine version must be specified. The new
            DB parameter group can be the default for that DB parameter group
            family.

        Example: `5.1.42`

        :type allow_major_version_upgrade: boolean
        :param allow_major_version_upgrade: Indicates that major version
            upgrades are allowed. Changing this parameter does not result in an
            outage and the change is asynchronously applied as soon as
            possible.
        Constraints: This parameter must be set to true when specifying a value
            for the EngineVersion parameter that is a different major version
            than the DB instance's current version.

        :type auto_minor_version_upgrade: boolean
        :param auto_minor_version_upgrade: Indicates that minor version
            upgrades will be applied automatically to the DB instance during
            the maintenance window. Changing this parameter does not result in
            an outage except in the following case and the change is
            asynchronously applied as soon as possible. An outage will result
            if this parameter is set to `True` during the maintenance window,
            and a newer minor version is available, and RDS has enabled auto
            patching for that engine version.

        :type iops: integer
        :param iops: The new Provisioned IOPS (I/O operations per second) value
            for the RDS instance. Changing this parameter does not result in an
            outage and the change is applied during the next maintenance window
            unless the `ApplyImmediately` parameter is set to `True` for this
            request.
        Default: Uses existing setting

        Constraints: Value supplied must be at least 10% greater than the
            current value. Values that are not at least 10% greater than the
            existing value are rounded up so that they are 10% greater than the
            current value.

        Type: Integer

        If you choose to migrate your DB instance from using standard storage
            to using Provisioned IOPS, or from using Provisioned IOPS to using
            standard storage, the process can take time. The duration of the
            migration depends on several factors such as database load, storage
            size, storage type (standard or Provisioned IOPS), amount of IOPS
            provisioned (if any), and the number of prior scale storage
            operations. Typical migration times are under 24 hours, but the
            process can take up to several days in some cases. During the
            migration, the DB instance will be available for use, but may
            experience performance degradation. While the migration takes
            place, nightly backups for the instance will be suspended. No other
            Amazon RDS operations can take place for the instance, including
            modifying the instance, rebooting the instance, deleting the
            instance, creating a read replica for the instance, and creating a
            DB snapshot of the instance.

        :type option_group_name: string
        :param option_group_name: Indicates that the DB instance should be
            associated with the specified option group. Changing this parameter
            does not result in an outage except in the following case and the
            change is applied during the next maintenance window unless the
            `ApplyImmediately` parameter is set to `True` for this request. If
            the parameter change results in an option group that enables OEM,
            this change can cause a brief (sub-second) period during which new
            connections are rejected but existing connections are not
            interrupted.
        Permanent options, such as the TDE option for Oracle Advanced Security
            TDE, cannot be removed from an option group, and that option group
            cannot be removed from a DB instance once it is associated with a
            DB instance

        :type new_db_instance_identifier: string
        :param new_db_instance_identifier:
        The new DB instance identifier for the DB instance when renaming a DB
            Instance. This value is stored as a lowercase string.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        """
    params = {'DBInstanceIdentifier': db_instance_identifier}
    if allocated_storage is not None:
        params['AllocatedStorage'] = allocated_storage
    if db_instance_class is not None:
        params['DBInstanceClass'] = db_instance_class
    if db_security_groups is not None:
        self.build_list_params(params, db_security_groups, 'DBSecurityGroups.member')
    if vpc_security_group_ids is not None:
        self.build_list_params(params, vpc_security_group_ids, 'VpcSecurityGroupIds.member')
    if apply_immediately is not None:
        params['ApplyImmediately'] = str(apply_immediately).lower()
    if master_user_password is not None:
        params['MasterUserPassword'] = master_user_password
    if db_parameter_group_name is not None:
        params['DBParameterGroupName'] = db_parameter_group_name
    if backup_retention_period is not None:
        params['BackupRetentionPeriod'] = backup_retention_period
    if preferred_backup_window is not None:
        params['PreferredBackupWindow'] = preferred_backup_window
    if preferred_maintenance_window is not None:
        params['PreferredMaintenanceWindow'] = preferred_maintenance_window
    if multi_az is not None:
        params['MultiAZ'] = str(multi_az).lower()
    if engine_version is not None:
        params['EngineVersion'] = engine_version
    if allow_major_version_upgrade is not None:
        params['AllowMajorVersionUpgrade'] = str(allow_major_version_upgrade).lower()
    if auto_minor_version_upgrade is not None:
        params['AutoMinorVersionUpgrade'] = str(auto_minor_version_upgrade).lower()
    if iops is not None:
        params['Iops'] = iops
    if option_group_name is not None:
        params['OptionGroupName'] = option_group_name
    if new_db_instance_identifier is not None:
        params['NewDBInstanceIdentifier'] = new_db_instance_identifier
    return self._make_request(action='ModifyDBInstance', verb='POST', path='/', params=params)