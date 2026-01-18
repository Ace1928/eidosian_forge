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
def modify_db_subnet_group(self, name, description=None, subnet_ids=None):
    """
        Modify a parameter group for your account.

        :type name: string
        :param name: The name of the new parameter group

        :type parameters: list of :class:`boto.rds.parametergroup.Parameter`
        :param parameters: The new parameters

        :rtype: :class:`boto.rds.parametergroup.ParameterGroup`
        :return: The newly created ParameterGroup
        """
    params = {'DBSubnetGroupName': name}
    if description is not None:
        params['DBSubnetGroupDescription'] = description
    if subnet_ids is not None:
        self.build_list_params(params, subnet_ids, 'SubnetIds.member')
    return self.get_object('ModifyDBSubnetGroup', params, DBSubnetGroup)