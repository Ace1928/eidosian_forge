import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def describe_orderable_cluster_options(self, cluster_version=None, node_type=None, max_records=None, marker=None):
    """
        Returns a list of orderable cluster options. Before you create
        a new cluster you can use this operation to find what options
        are available, such as the EC2 Availability Zones (AZ) in the
        specific AWS region that you can specify, and the node types
        you can request. The node types differ by available storage,
        memory, CPU and price. With the cost involved you might want
        to obtain a list of cluster options in the specific region and
        specify values when creating a cluster. For more information
        about managing clusters, go to `Amazon Redshift Clusters`_ in
        the Amazon Redshift Management Guide

        :type cluster_version: string
        :param cluster_version: The version filter value. Specify this
            parameter to show only the available offerings matching the
            specified version.
        Default: All versions.

        Constraints: Must be one of the version returned from
            DescribeClusterVersions.

        :type node_type: string
        :param node_type: The node type filter value. Specify this parameter to
            show only the available offerings matching the specified node type.

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
            DescribeOrderableClusterOptions request exceed the value specified
            in `MaxRecords`, AWS returns a value in the `Marker` field of the
            response. You can retrieve the next set of response records by
            providing the returned marker value in the `Marker` parameter and
            retrying the request.

        """
    params = {}
    if cluster_version is not None:
        params['ClusterVersion'] = cluster_version
    if node_type is not None:
        params['NodeType'] = node_type
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeOrderableClusterOptions', verb='POST', path='/', params=params)