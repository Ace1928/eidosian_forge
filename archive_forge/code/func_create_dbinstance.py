import urllib
from boto.connection import AWSQueryConnection
from boto.rds.dbinstance import DBInstance
from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.rds.optiongroup  import OptionGroup, OptionGroupOption
from boto.rds.parametergroup import ParameterGroup
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds.event import Event
from boto.rds.regioninfo import RDSRegionInfo
from boto.rds.dbsubnetgroup import DBSubnetGroup
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.regioninfo import get_regions
from boto.regioninfo import connect
from boto.rds.logfile import LogFile, LogFileObject
def create_dbinstance(self, id, allocated_storage, instance_class, master_username, master_password, port=3306, engine='MySQL5.1', db_name=None, param_group=None, security_groups=None, availability_zone=None, preferred_maintenance_window=None, backup_retention_period=None, preferred_backup_window=None, multi_az=False, engine_version=None, auto_minor_version_upgrade=True, character_set_name=None, db_subnet_group_name=None, license_model=None, option_group_name=None, iops=None, vpc_security_groups=None):
    """
        Create a new DBInstance.

        :type id: str
        :param id: Unique identifier for the new instance.
                   Must contain 1-63 alphanumeric characters.
                   First character must be a letter.
                   May not end with a hyphen or contain two consecutive hyphens

        :type allocated_storage: int
        :param allocated_storage: Initially allocated storage size, in GBs.
                                  Valid values are depending on the engine value.

                                  * MySQL = 5--3072
                                  * oracle-se1 = 10--3072
                                  * oracle-se = 10--3072
                                  * oracle-ee = 10--3072
                                  * sqlserver-ee = 200--1024
                                  * sqlserver-se = 200--1024
                                  * sqlserver-ex = 30--1024
                                  * sqlserver-web = 30--1024
                                  * postgres = 5--3072

        :type instance_class: str
        :param instance_class: The compute and memory capacity of
                               the DBInstance. Valid values are:

                               * db.t1.micro
                               * db.m1.small
                               * db.m1.medium
                               * db.m1.large
                               * db.m1.xlarge
                               * db.m2.xlarge
                               * db.m2.2xlarge
                               * db.m2.4xlarge

        :type engine: str
        :param engine: Name of database engine. Defaults to MySQL but can be;

                       * MySQL
                       * oracle-se1
                       * oracle-se
                       * oracle-ee
                       * sqlserver-ee
                       * sqlserver-se
                       * sqlserver-ex
                       * sqlserver-web
                       * postgres

        :type master_username: str
        :param master_username: Name of master user for the DBInstance.

                                * MySQL must be;
                                  - 1--16 alphanumeric characters
                                  - first character must be a letter
                                  - cannot be a reserved MySQL word

                                * Oracle must be:
                                  - 1--30 alphanumeric characters
                                  - first character must be a letter
                                  - cannot be a reserved Oracle word

                                * SQL Server must be:
                                  - 1--128 alphanumeric characters
                                  - first character must be a letter
                                  - cannot be a reserver SQL Server word

        :type master_password: str
        :param master_password: Password of master user for the DBInstance.

                                * MySQL must be 8--41 alphanumeric characters

                                * Oracle must be 8--30 alphanumeric characters

                                * SQL Server must be 8--128 alphanumeric characters.

        :type port: int
        :param port: Port number on which database accepts connections.
                     Valid values [1115-65535].

                     * MySQL defaults to 3306

                     * Oracle defaults to 1521

                     * SQL Server defaults to 1433 and _cannot_ be 1434, 3389,
                       47001, 49152, and 49152 through 49156.

                     * PostgreSQL defaults to 5432

        :type db_name: str
        :param db_name: * MySQL:
                          Name of a database to create when the DBInstance
                          is created. Default is to create no databases.

                          Must contain 1--64 alphanumeric characters and cannot
                          be a reserved MySQL word.

                        * Oracle:
                          The Oracle System ID (SID) of the created DB instances.
                          Default is ORCL. Cannot be longer than 8 characters.

                        * SQL Server:
                          Not applicable and must be None.

                        * PostgreSQL:
                          Name of a database to create when the DBInstance
                          is created. Default is to create no databases.

                          Must contain 1--63 alphanumeric characters. Must
                          begin with a letter or an underscore. Subsequent
                          characters can be letters, underscores, or digits (0-9)
                          and cannot be a reserved PostgreSQL word.

        :type param_group: str or ParameterGroup object
        :param param_group: Name of DBParameterGroup or ParameterGroup instance
                            to associate with this DBInstance.  If no groups are
                            specified no parameter groups will be used.

        :type security_groups: list of str or list of DBSecurityGroup objects
        :param security_groups: List of names of DBSecurityGroup to
            authorize on this DBInstance.

        :type availability_zone: str
        :param availability_zone: Name of the availability zone to place
                                  DBInstance into.

        :type preferred_maintenance_window: str
        :param preferred_maintenance_window: The weekly time range (in UTC)
                                             during which maintenance can occur.
                                             Default is Sun:05:00-Sun:09:00

        :type backup_retention_period: int
        :param backup_retention_period: The number of days for which automated
                                        backups are retained.  Setting this to
                                        zero disables automated backups.

        :type preferred_backup_window: str
        :param preferred_backup_window: The daily time range during which
                                        automated backups are created (if
                                        enabled).  Must be in h24:mi-hh24:mi
                                        format (UTC).

        :type multi_az: bool
        :param multi_az: If True, specifies the DB Instance will be
                         deployed in multiple availability zones.

                         For Microsoft SQL Server, must be set to false. You cannot set
                         the AvailabilityZone parameter if the MultiAZ parameter is
                         set to true.

        :type engine_version: str
        :param engine_version: The version number of the database engine to use.

                               * MySQL format example: 5.1.42

                               * Oracle format example: 11.2.0.2.v2

                               * SQL Server format example: 10.50.2789.0.v1

                               * PostgreSQL format example: 9.3

        :type auto_minor_version_upgrade: bool
        :param auto_minor_version_upgrade: Indicates that minor engine
                                           upgrades will be applied
                                           automatically to the Read Replica
                                           during the maintenance window.
                                           Default is True.
        :type character_set_name: str
        :param character_set_name: For supported engines, indicates that the DB Instance
                                   should be associated with the specified CharacterSet.

        :type db_subnet_group_name: str
        :param db_subnet_group_name: A DB Subnet Group to associate with this DB Instance.
                                     If there is no DB Subnet Group, then it is a non-VPC DB
                                     instance.

        :type license_model: str
        :param license_model: License model information for this DB Instance.

                              Valid values are;
                              - license-included
                              - bring-your-own-license
                              - general-public-license

                              All license types are not supported on all engines.

        :type option_group_name: str
        :param option_group_name: Indicates that the DB Instance should be associated
                                  with the specified option group.

        :type iops: int
        :param iops:  The amount of IOPS (input/output operations per second) to Provisioned
                      for the DB Instance. Can be modified at a later date.

                      Must scale linearly. For every 1000 IOPS provision, you must allocated
                      100 GB of storage space. This scales up to 1 TB / 10 000 IOPS for MySQL
                      and Oracle. MSSQL is limited to 700 GB / 7 000 IOPS.

                      If you specify a value, it must be at least 1000 IOPS and you must
                      allocate 100 GB of storage.

        :type vpc_security_groups: list of str or a VPCSecurityGroupMembership object
        :param vpc_security_groups: List of VPC security group ids or a list of
            VPCSecurityGroupMembership objects this DBInstance should be a member of

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The new db instance.
        """
    params = {'AllocatedStorage': allocated_storage, 'AutoMinorVersionUpgrade': str(auto_minor_version_upgrade).lower() if auto_minor_version_upgrade else None, 'AvailabilityZone': availability_zone, 'BackupRetentionPeriod': backup_retention_period, 'CharacterSetName': character_set_name, 'DBInstanceClass': instance_class, 'DBInstanceIdentifier': id, 'DBName': db_name, 'DBParameterGroupName': param_group.name if isinstance(param_group, ParameterGroup) else param_group, 'DBSubnetGroupName': db_subnet_group_name, 'Engine': engine, 'EngineVersion': engine_version, 'Iops': iops, 'LicenseModel': license_model, 'MasterUsername': master_username, 'MasterUserPassword': master_password, 'MultiAZ': str(multi_az).lower() if multi_az else None, 'OptionGroupName': option_group_name, 'Port': port, 'PreferredBackupWindow': preferred_backup_window, 'PreferredMaintenanceWindow': preferred_maintenance_window}
    if security_groups:
        l = []
        for group in security_groups:
            if isinstance(group, DBSecurityGroup):
                l.append(group.name)
            else:
                l.append(group)
        self.build_list_params(params, l, 'DBSecurityGroups.member')
    if vpc_security_groups:
        l = []
        for vpc_grp in vpc_security_groups:
            if isinstance(vpc_grp, VPCSecurityGroupMembership):
                l.append(vpc_grp.vpc_group)
            else:
                l.append(vpc_grp)
        self.build_list_params(params, l, 'VpcSecurityGroupIds.member')
    for k, v in list(params.items()):
        if v is None:
            del params[k]
    return self.get_object('CreateDBInstance', params, DBInstance)