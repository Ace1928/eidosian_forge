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
def delete_dbsecurity_group(self, name):
    """
        Delete a DBSecurityGroup from your account.

        :type key_name: string
        :param key_name: The name of the DBSecurityGroup to delete
        """
    params = {'DBSecurityGroupName': name}
    return self.get_status('DeleteDBSecurityGroup', params)