import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_db_parameter_groups(self, db_parameter_group_name=None, filters=None, max_records=None, marker=None):
    """
        Returns a list of `DBParameterGroup` descriptions. If a
        `DBParameterGroupName` is specified, the list will contain
        only the description of the specified DB parameter group.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of a specific DB parameter group to return details for.

        Constraints:


        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type filters: list
        :param filters:

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a pagination token called a marker is included in the
            response so that the remaining results may be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            `DescribeDBParameterGroups` request. If this parameter is
            specified, the response includes only records beyond the marker, up
            to the value specified by `MaxRecords`.

        """
    params = {}
    if db_parameter_group_name is not None:
        params['DBParameterGroupName'] = db_parameter_group_name
    if filters is not None:
        self.build_complex_list_params(params, filters, 'Filters.member', ('FilterName', 'FilterValue'))
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeDBParameterGroups', verb='POST', path='/', params=params)