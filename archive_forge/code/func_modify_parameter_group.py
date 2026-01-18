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
def modify_parameter_group(self, name, parameters=None):
    """
        Modify a ParameterGroup for your account.

        :type name: string
        :param name: The name of the new ParameterGroup

        :type parameters: list of :class:`boto.rds.parametergroup.Parameter`
        :param parameters: The new parameters

        :rtype: :class:`boto.rds.parametergroup.ParameterGroup`
        :return: The newly created ParameterGroup
        """
    params = {'DBParameterGroupName': name}
    for i in range(0, len(parameters)):
        parameter = parameters[i]
        parameter.merge(params, i + 1)
    return self.get_list('ModifyDBParameterGroup', params, ParameterGroup, verb='POST')