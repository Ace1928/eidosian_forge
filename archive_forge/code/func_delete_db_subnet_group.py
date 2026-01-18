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
def delete_db_subnet_group(self, name):
    """
        Delete a Database Subnet Group.

        :type name: string
        :param name: The identifier of the db_subnet_group to delete

        :rtype: :class:`boto.rds.dbsubnetgroup.DBSubnetGroup`
        :return: The deleted db_subnet_group.
        """
    params = {'DBSubnetGroupName': name}
    return self.get_object('DeleteDBSubnetGroup', params, DBSubnetGroup)