import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_log_files(self, db_instance_identifier, filename_contains=None, file_last_written=None, file_size=None, max_records=None, marker=None):
    """
        Returns a list of DB log files for the DB instance.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The customer-assigned name of the DB instance that contains the log
            files you want to list.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type filename_contains: string
        :param filename_contains: Filters the available log files for log file
            names that contain the specified string.

        :type file_last_written: long
        :param file_last_written: Filters the available log files for files
            written since the specified date, in POSIX timestamp format.

        :type file_size: long
        :param file_size: Filters the available log files for files larger than
            the specified size.

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified MaxRecords
            value, a pagination token called a marker is included in the
            response so that the remaining results can be retrieved.

        :type marker: string
        :param marker: The pagination token provided in the previous request.
            If this parameter is specified the response includes only records
            beyond the marker, up to MaxRecords.

        """
    params = {'DBInstanceIdentifier': db_instance_identifier}
    if filename_contains is not None:
        params['FilenameContains'] = filename_contains
    if file_last_written is not None:
        params['FileLastWritten'] = file_last_written
    if file_size is not None:
        params['FileSize'] = file_size
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBLogFiles', verb='POST', path='/', params=params)