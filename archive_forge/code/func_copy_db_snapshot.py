import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def copy_db_snapshot(self, source_db_snapshot_identifier, target_db_snapshot_identifier, tags=None):
    """
        Copies the specified DBSnapshot. The source DBSnapshot must be
        in the "available" state.

        :type source_db_snapshot_identifier: string
        :param source_db_snapshot_identifier: The identifier for the source DB
            snapshot.
        Constraints:


        + Must be the identifier for a valid system snapshot in the "available"
              state.


        Example: `rds:mydb-2012-04-02-00-01`

        :type target_db_snapshot_identifier: string
        :param target_db_snapshot_identifier: The identifier for the copied
            snapshot.
        Constraints:


        + Cannot be null, empty, or blank
        + Must contain from 1 to 255 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens


        Example: `my-db-snapshot`

        :type tags: list
        :param tags: A list of tags. Tags must be passed as tuples in the form
            [('key1', 'valueForKey1'), ('key2', 'valueForKey2')]
        """
    params = {'SourceDBSnapshotIdentifier': source_db_snapshot_identifier, 'TargetDBSnapshotIdentifier': target_db_snapshot_identifier}
    if tags is not None:
        self.build_complex_list_params(params, tags, 'Tags.member', ('Key', 'Value'))
    return self._make_request(action='CopyDBSnapshot', verb='POST', path='/', params=params)