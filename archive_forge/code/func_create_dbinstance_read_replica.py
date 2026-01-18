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
def create_dbinstance_read_replica(self, id, source_id, instance_class=None, port=3306, availability_zone=None, auto_minor_version_upgrade=None):
    """
        Create a new DBInstance Read Replica.

        :type id: str
        :param id: Unique identifier for the new instance.
                   Must contain 1-63 alphanumeric characters.
                   First character must be a letter.
                   May not end with a hyphen or contain two consecutive hyphens

        :type source_id: str
        :param source_id: Unique identifier for the DB Instance for which this
                          DB Instance will act as a Read Replica.

        :type instance_class: str
        :param instance_class: The compute and memory capacity of the
                               DBInstance.  Default is to inherit from
                               the source DB Instance.

                               Valid values are:

                               * db.m1.small
                               * db.m1.large
                               * db.m1.xlarge
                               * db.m2.xlarge
                               * db.m2.2xlarge
                               * db.m2.4xlarge

        :type port: int
        :param port: Port number on which database accepts connections.
                     Default is to inherit from source DB Instance.
                     Valid values [1115-65535].  Defaults to 3306.

        :type availability_zone: str
        :param availability_zone: Name of the availability zone to place
                                  DBInstance into.

        :type auto_minor_version_upgrade: bool
        :param auto_minor_version_upgrade: Indicates that minor engine
                                           upgrades will be applied
                                           automatically to the Read Replica
                                           during the maintenance window.
                                           Default is to inherit this value
                                           from the source DB Instance.

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The new db instance.
        """
    params = {'DBInstanceIdentifier': id, 'SourceDBInstanceIdentifier': source_id}
    if instance_class:
        params['DBInstanceClass'] = instance_class
    if port:
        params['Port'] = port
    if availability_zone:
        params['AvailabilityZone'] = availability_zone
    if auto_minor_version_upgrade is not None:
        if auto_minor_version_upgrade is True:
            params['AutoMinorVersionUpgrade'] = 'true'
        else:
            params['AutoMinorVersionUpgrade'] = 'false'
    return self.get_object('CreateDBInstanceReadReplica', params, DBInstance)