import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def download_db_log_file_portion(self, db_instance_identifier, log_file_name, marker=None, number_of_lines=None):
    """
        Downloads the last line of the specified log file.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The customer-assigned name of the DB instance that contains the log
            files you want to list.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type log_file_name: string
        :param log_file_name: The name of the log file to be downloaded.

        :type marker: string
        :param marker: The pagination token provided in the previous request.
            If this parameter is specified the response includes only records
            beyond the marker, up to MaxRecords.

        :type number_of_lines: integer
        :param number_of_lines: The number of lines remaining to be downloaded.

        """
    params = {'DBInstanceIdentifier': db_instance_identifier, 'LogFileName': log_file_name}
    if marker is not None:
        params['Marker'] = marker
    if number_of_lines is not None:
        params['NumberOfLines'] = number_of_lines
    return self._make_request(action='DownloadDBLogFilePortion', verb='POST', path='/', params=params)