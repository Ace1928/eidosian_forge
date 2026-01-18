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
def reboot_dbinstance(self, id):
    """
        Reboot DBInstance.

        :type id: str
        :param id: Unique identifier of the instance.

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The rebooting db instance.
        """
    params = {'DBInstanceIdentifier': id}
    return self.get_object('RebootDBInstance', params, DBInstance)