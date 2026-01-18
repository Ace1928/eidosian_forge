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
def create_dbsecurity_group(self, name, description=None):
    """
        Create a new security group for your account.
        This will create the security group within the region you
        are currently connected to.

        :type name: string
        :param name: The name of the new security group

        :type description: string
        :param description: The description of the new security group

        :rtype: :class:`boto.rds.dbsecuritygroup.DBSecurityGroup`
        :return: The newly created DBSecurityGroup
        """
    params = {'DBSecurityGroupName': name}
    if description:
        params['DBSecurityGroupDescription'] = description
    group = self.get_object('CreateDBSecurityGroup', params, DBSecurityGroup)
    group.name = name
    group.description = description
    return group