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
def get_all_dbparameters(self, groupname, source=None, max_records=None, marker=None):
    """
        Get all parameters associated with a ParameterGroup

        :type groupname: str
        :param groupname: The name of the DBParameter group to retrieve.

        :type source: str
        :param source: Specifies which parameters to return.
                       If not specified, all parameters will be returned.
                       Valid values are: user|system|engine-default

        :type max_records: int
        :param max_records: The maximum number of records to be returned.
                            If more results are available, a MoreToken will
                            be returned in the response that can be used to
                            retrieve additional records.  Default is 100.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :rtype: :class:`boto.ec2.parametergroup.ParameterGroup`
        :return: The ParameterGroup
        """
    params = {'DBParameterGroupName': groupname}
    if source:
        params['Source'] = source
    if max_records:
        params['MaxRecords'] = max_records
    if marker:
        params['Marker'] = marker
    pg = self.get_object('DescribeDBParameters', params, ParameterGroup)
    pg.name = groupname
    return pg