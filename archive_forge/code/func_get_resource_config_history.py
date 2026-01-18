import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def get_resource_config_history(self, resource_type, resource_id, later_time=None, earlier_time=None, chronological_order=None, limit=None, next_token=None):
    """
        Returns a list of configuration items for the specified
        resource. The list contains details about each state of the
        resource during the specified time interval. You can specify a
        `limit` on the number of results returned on the page. If a
        limit is specified, a `nextToken` is returned as part of the
        result that you can use to continue this request.

        :type resource_type: string
        :param resource_type: The resource type.

        :type resource_id: string
        :param resource_id: The ID of the resource (for example., `sg-xxxxxx`).

        :type later_time: timestamp
        :param later_time: The time stamp that indicates a later time. If not
            specified, current time is taken.

        :type earlier_time: timestamp
        :param earlier_time: The time stamp that indicates an earlier time. If
            not specified, the action returns paginated results that contain
            configuration items that start from when the first configuration
            item was recorded.

        :type chronological_order: string
        :param chronological_order: The chronological order for configuration
            items listed. By default the results are listed in reverse
            chronological order.

        :type limit: integer
        :param limit: The maximum number of configuration items returned in
            each page. The default is 10. You cannot specify a limit greater
            than 100.

        :type next_token: string
        :param next_token: An optional parameter used for pagination of the
            results.

        """
    params = {'resourceType': resource_type, 'resourceId': resource_id}
    if later_time is not None:
        params['laterTime'] = later_time
    if earlier_time is not None:
        params['earlierTime'] = earlier_time
    if chronological_order is not None:
        params['chronologicalOrder'] = chronological_order
    if limit is not None:
        params['limit'] = limit
    if next_token is not None:
        params['nextToken'] = next_token
    return self.make_request(action='GetResourceConfigHistory', body=json.dumps(params))