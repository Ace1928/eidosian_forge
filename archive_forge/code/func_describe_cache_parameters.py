import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_cache_parameters(self, cache_parameter_group_name, source=None, max_records=None, marker=None):
    """
        The DescribeCacheParameters operation returns the detailed
        parameter list for a particular cache parameter group.

        :type cache_parameter_group_name: string
        :param cache_parameter_group_name: The name of a specific cache
            parameter group to return details for.

        :type source: string
        :param source: The parameter types to return.
        Valid values: `user` | `system` | `engine-default`

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more records exist than the specified `MaxRecords`
            value, a marker is included in the response so that the remaining
            results can be retrieved.
        Default: 100

        Constraints: minimum 20; maximum 100.

        :type marker: string
        :param marker: An optional marker returned from a prior request. Use
            this marker for pagination of results from this operation. If this
            parameter is specified, the response includes only records beyond
            the marker, up to the value specified by MaxRecords .

        """
    params = {'CacheParameterGroupName': cache_parameter_group_name}
    if source is not None:
        params['Source'] = source
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeCacheParameters', verb='POST', path='/', params=params)