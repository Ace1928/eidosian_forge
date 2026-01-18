import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def get_log_events(self, log_group_name, log_stream_name, start_time=None, end_time=None, next_token=None, limit=None, start_from_head=None):
    """
        Retrieves log events from the specified log stream. You can
        provide an optional time range to filter the results on the
        event `timestamp`.

        By default, this operation returns as much log events as can
        fit in a response size of 1MB, up to 10,000 log events. The
        response will always include a `nextForwardToken` and a
        `nextBackwardToken` in the response body. You can use any of
        these tokens in subsequent `GetLogEvents` requests to paginate
        through events in either forward or backward direction. You
        can also limit the number of log events returned in the
        response by specifying the `limit` parameter in the request.

        :type log_group_name: string
        :param log_group_name:

        :type log_stream_name: string
        :param log_stream_name:

        :type start_time: long
        :param start_time: A point in time expressed as the number milliseconds
            since Jan 1, 1970 00:00:00 UTC.

        :type end_time: long
        :param end_time: A point in time expressed as the number milliseconds
            since Jan 1, 1970 00:00:00 UTC.

        :type next_token: string
        :param next_token: A string token used for pagination that points to
            the next page of results. It must be a value obtained from the
            `nextForwardToken` or `nextBackwardToken` fields in the response of
            the previous `GetLogEvents` request.

        :type limit: integer
        :param limit: The maximum number of log events returned in the
            response. If you don't specify a value, the request would return as
            much log events as can fit in a response size of 1MB, up to 10,000
            log events.

        :type start_from_head: boolean
        :param start_from_head:

        """
    params = {'logGroupName': log_group_name, 'logStreamName': log_stream_name}
    if start_time is not None:
        params['startTime'] = start_time
    if end_time is not None:
        params['endTime'] = end_time
    if next_token is not None:
        params['nextToken'] = next_token
    if limit is not None:
        params['limit'] = limit
    if start_from_head is not None:
        params['startFromHead'] = start_from_head
    return self.make_request(action='GetLogEvents', body=json.dumps(params))