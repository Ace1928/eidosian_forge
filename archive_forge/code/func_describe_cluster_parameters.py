import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_cluster_parameters(self, parameter_group_name, source=None, max_records=None, marker=None):
    """
        Returns a detailed list of parameters contained within the
        specified Amazon Redshift parameter group. For each parameter
        the response includes information such as parameter name,
        description, data type, value, whether the parameter value is
        modifiable, and so on.

        You can specify source filter to retrieve parameters of only
        specific type. For example, to retrieve parameters that were
        modified by a user action such as from
        ModifyClusterParameterGroup, you can specify source equal to
        user .

        For more information about managing parameter groups, go to
        `Amazon Redshift Parameter Groups`_ in the Amazon Redshift
        Management Guide .

        :type parameter_group_name: string
        :param parameter_group_name: The name of a cluster parameter group for
            which to return details.

        :type source: string
        :param source: The parameter types to return. Specify `user` to show
            parameters that are different form the default. Similarly, specify
            `engine-default` to show parameters that are the same as the
            default parameter group.
        Default: All parameter types returned.

        Valid Values: `user` | `engine-default`

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
            DescribeClusterParameters request exceed the value specified in
            `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {'ParameterGroupName': parameter_group_name}
    if source is not None:
        params['Source'] = source
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeClusterParameters', verb='POST', path='/', params=params)