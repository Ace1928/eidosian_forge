import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def describe_metric_filters(self, log_group_name, filter_name_prefix=None, next_token=None, limit=None):
    """
        Returns all the metrics filters associated with the specified
        log group. The list returned in the response is ASCII-sorted
        by filter name.

        By default, this operation returns up to 50 metric filters. If
        there are more metric filters to list, the response would
        contain a `nextToken` value in the response body. You can also
        limit the number of metric filters returned in the response by
        specifying the `limit` parameter in the request.

        :type log_group_name: string
        :param log_group_name:

        :type filter_name_prefix: string
        :param filter_name_prefix: The name of the metric filter.

        :type next_token: string
        :param next_token: A string token used for pagination that points to
            the next page of results. It must be a value obtained from the
            response of the previous `DescribeMetricFilters` request.

        :type limit: integer
        :param limit: The maximum number of items returned in the response. If
            you don't specify a value, the request would return up to 50 items.

        """
    params = {'logGroupName': log_group_name}
    if filter_name_prefix is not None:
        params['filterNamePrefix'] = filter_name_prefix
    if next_token is not None:
        params['nextToken'] = next_token
    if limit is not None:
        params['limit'] = limit
    return self.make_request(action='DescribeMetricFilters', body=json.dumps(params))