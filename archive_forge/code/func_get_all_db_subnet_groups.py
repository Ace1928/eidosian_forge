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
def get_all_db_subnet_groups(self, name=None, max_records=None, marker=None):
    """
        Retrieve all the DBSubnetGroups in your account.

        :type name: str
        :param name: DBSubnetGroup name If supplied, only information about
                     this DBSubnetGroup will be returned. Otherwise, info
                     about all DBSubnetGroups will be returned.

        :type max_records: int
        :param max_records: The maximum number of records to be returned.
                            If more results are available, a Token will be
                            returned in the response that can be used to
                            retrieve additional records.  Default is 100.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :rtype: list
        :return: A list of :class:`boto.rds.dbsubnetgroup.DBSubnetGroup`
        """
    params = dict()
    if name is not None:
        params['DBSubnetGroupName'] = name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self.get_list('DescribeDBSubnetGroups', params, [('DBSubnetGroup', DBSubnetGroup)])