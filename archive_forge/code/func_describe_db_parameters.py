import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_parameters(self, db_parameter_group_name, source=None, max_records=None, marker=None):
    """
        Returns the detailed parameter list for a particular DB
        parameter group.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of a specific DB parameter group to return details for.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type source: string
        :param source: The parameter types to return.
        Default: All parameter types returned

        Valid Values: `user | system | engine-default`

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a pagination token called a marker is included in the
            response so that the remaining results may be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            `DescribeDBParameters` request. If this parameter is specified, the
            response includes only records beyond the marker, up to the value
            specified by `MaxRecords`.

        """
    params = {'DBParameterGroupName': db_parameter_group_name}
    if source is not None:
        params['Source'] = source
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBParameters', verb='POST', path='/', params=params)