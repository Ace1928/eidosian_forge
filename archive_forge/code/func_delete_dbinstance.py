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
def delete_dbinstance(self, id, skip_final_snapshot=False, final_snapshot_id=''):
    """
        Delete an existing DBInstance.

        :type id: str
        :param id: Unique identifier for the new instance.

        :type skip_final_snapshot: bool
        :param skip_final_snapshot: This parameter determines whether a final
                                    db snapshot is created before the instance
                                    is deleted.  If True, no snapshot
                                    is created.  If False, a snapshot
                                    is created before deleting the instance.

        :type final_snapshot_id: str
        :param final_snapshot_id: If a final snapshot is requested, this
                                  is the identifier used for that snapshot.

        :rtype: :class:`boto.rds.dbinstance.DBInstance`
        :return: The deleted db instance.
        """
    params = {'DBInstanceIdentifier': id}
    if skip_final_snapshot:
        params['SkipFinalSnapshot'] = 'true'
    else:
        params['SkipFinalSnapshot'] = 'false'
        params['FinalDBSnapshotIdentifier'] = final_snapshot_id
    return self.get_object('DeleteDBInstance', params, DBInstance)