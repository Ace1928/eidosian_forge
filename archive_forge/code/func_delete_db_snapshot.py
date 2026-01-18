import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def delete_db_snapshot(self, db_snapshot_identifier):
    """
        Deletes a DBSnapshot.
        The DBSnapshot must be in the `available` state to be deleted.

        :type db_snapshot_identifier: string
        :param db_snapshot_identifier: The DBSnapshot identifier.
        Constraints: Must be the name of an existing DB snapshot in the
            `available` state.

        """
    params = {'DBSnapshotIdentifier': db_snapshot_identifier}
    return self._make_request(action='DeleteDBSnapshot', verb='POST', path='/', params=params)