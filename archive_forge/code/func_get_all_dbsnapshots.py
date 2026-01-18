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
def get_all_dbsnapshots(self, snapshot_id=None, instance_id=None, max_records=None, marker=None):
    """
        Get information about DB Snapshots.

        :type snapshot_id: str
        :param snapshot_id: The unique identifier of an RDS snapshot.
                            If not provided, all RDS snapshots will be returned.

        :type instance_id: str
        :param instance_id: The identifier of a DBInstance.  If provided,
                            only the DBSnapshots related to that instance will
                            be returned.
                            If not provided, all RDS snapshots will be returned.

        :type max_records: int
        :param max_records: The maximum number of records to be returned.
                            If more results are available, a MoreToken will
                            be returned in the response that can be used to
                            retrieve additional records.  Default is 100.

        :type marker: str
        :param marker: The marker provided by a previous request.

        :rtype: list
        :return: A list of :class:`boto.rds.dbsnapshot.DBSnapshot`
        """
    params = {}
    if snapshot_id:
        params['DBSnapshotIdentifier'] = snapshot_id
    if instance_id:
        params['DBInstanceIdentifier'] = instance_id
    if max_records:
        params['MaxRecords'] = max_records
    if marker:
        params['Marker'] = marker
    return self.get_list('DescribeDBSnapshots', params, [('DBSnapshot', DBSnapshot)])