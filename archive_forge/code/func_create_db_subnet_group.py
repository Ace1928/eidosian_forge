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
def create_db_subnet_group(self, name, desc, subnet_ids):
    """
        Create a new Database Subnet Group.

        :type name: string
        :param name: The identifier for the db_subnet_group

        :type desc: string
        :param desc: A description of the db_subnet_group

        :type subnet_ids: list
        :param subnets: A list of the subnet identifiers to include in the
                        db_subnet_group

        :rtype: :class:`boto.rds.dbsubnetgroup.DBSubnetGroup
        :return: the created db_subnet_group
        """
    params = {'DBSubnetGroupName': name, 'DBSubnetGroupDescription': desc}
    self.build_list_params(params, subnet_ids, 'SubnetIds.member')
    return self.get_object('CreateDBSubnetGroup', params, DBSubnetGroup)