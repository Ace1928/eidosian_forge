import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def describe_log_groups(self, log_group_name_prefix=None, next_token=None, limit=None):
    """
        Returns all the log groups that are associated with the AWS
        account making the request. The list returned in the response
        is ASCII-sorted by log group name.

        By default, this operation returns up to 50 log groups. If
        there are more log groups to list, the response would contain
        a `nextToken` value in the response body. You can also limit
        the number of log groups returned in the response by
        specifying the `limit` parameter in the request.

        :type log_group_name_prefix: string
        :param log_group_name_prefix:

        :type next_token: string
        :param next_token: A string token used for pagination that points to
            the next page of results. It must be a value obtained from the
            response of the previous `DescribeLogGroups` request.

        :type limit: integer
        :param limit: The maximum number of items returned in the response. If
            you don't specify a value, the request would return up to 50 items.

        """
    params = {}
    if log_group_name_prefix is not None:
        params['logGroupNamePrefix'] = log_group_name_prefix
    if next_token is not None:
        params['nextToken'] = next_token
    if limit is not None:
        params['limit'] = limit
    return self.make_request(action='DescribeLogGroups', body=json.dumps(params))