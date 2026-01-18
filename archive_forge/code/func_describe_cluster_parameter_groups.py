import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_cluster_parameter_groups(self, parameter_group_name=None, max_records=None, marker=None):
    """
        Returns a list of Amazon Redshift parameter groups, including
        parameter groups you created and the default parameter group.
        For each parameter group, the response includes the parameter
        group name, description, and parameter group family name. You
        can optionally specify a name to retrieve the description of a
        specific parameter group.

        For more information about managing parameter groups, go to
        `Amazon Redshift Parameter Groups`_ in the Amazon Redshift
        Management Guide .

        :type parameter_group_name: string
        :param parameter_group_name: The name of a specific parameter group for
            which to return details. By default, details about all parameter
            groups and the default parameter group are returned.

        :type max_records: integer
        :param max_records: The maximum number of response records to return in
            each call. If the number of remaining response records exceeds the
            specified `MaxRecords` value, a value is returned in a `marker`
            field of the response. You can retrieve the next set of records by
            retrying the command with the returned marker value.
        Default: `100`

        Constraints: minimum 20, maximum 100.

        :type marker: string
        :param marker: An optional parameter that specifies the starting point
            to return a set of response records. When the results of a
            DescribeClusterParameterGroups request exceed the value specified
            in `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if parameter_group_name is not None:
        params['ParameterGroupName'] = parameter_group_name
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeClusterParameterGroups', verb='POST', path='/', params=params)