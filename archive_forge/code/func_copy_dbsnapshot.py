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
def copy_dbsnapshot(self, source_snapshot_id, target_snapshot_id):
    """
        Copies the specified DBSnapshot.

        :type source_snapshot_id: string
        :param source_snapshot_id: The identifier for the source DB snapshot.

        :type target_snapshot_id: string
        :param target_snapshot_id: The identifier for the copied snapshot.

        :rtype: :class:`boto.rds.dbsnapshot.DBSnapshot`
        :return: The newly created DBSnapshot.
        """
    params = {'SourceDBSnapshotIdentifier': source_snapshot_id, 'TargetDBSnapshotIdentifier': target_snapshot_id}
    return self.get_object('CopyDBSnapshot', params, DBSnapshot)