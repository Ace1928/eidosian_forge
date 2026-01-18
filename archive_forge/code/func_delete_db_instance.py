import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def delete_db_instance(self, db_instance_identifier, skip_final_snapshot=None, final_db_snapshot_identifier=None):
    """
        The DeleteDBInstance action deletes a previously provisioned
        DB instance. A successful response from the web service
        indicates the request was received correctly. When you delete
        a DB instance, all automated backups for that instance are
        deleted and cannot be recovered. Manual DB snapshots of the DB
        instance to be deleted are not deleted.

        If a final DB snapshot is requested the status of the RDS
        instance will be "deleting" until the DB snapshot is created.
        The API action `DescribeDBInstance` is used to monitor the
        status of this operation. The action cannot be canceled or
        reverted once submitted.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The DB instance identifier for the DB instance to be deleted. This
            parameter isn't case sensitive.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type skip_final_snapshot: boolean
        :param skip_final_snapshot: Determines whether a final DB snapshot is
            created before the DB instance is deleted. If `True` is specified,
            no DBSnapshot is created. If false is specified, a DB snapshot is
            created before the DB instance is deleted.
        The FinalDBSnapshotIdentifier parameter must be specified if
            SkipFinalSnapshot is `False`.

        Default: `False`

        :type final_db_snapshot_identifier: string
        :param final_db_snapshot_identifier:
        The DBSnapshotIdentifier of the new DBSnapshot created when
            SkipFinalSnapshot is set to `False`.

        Specifying this parameter and also setting the SkipFinalShapshot
            parameter to true results in an error.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        """
    params = {'DBInstanceIdentifier': db_instance_identifier}
    if skip_final_snapshot is not None:
        params['SkipFinalSnapshot'] = str(skip_final_snapshot).lower()
    if final_db_snapshot_identifier is not None:
        params['FinalDBSnapshotIdentifier'] = final_db_snapshot_identifier
    return self._make_request(action='DeleteDBInstance', verb='POST', path='/', params=params)