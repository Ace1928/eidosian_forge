import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def put_log_events(self, log_group_name, log_stream_name, log_events, sequence_token=None):
    """
        Uploads a batch of log events to the specified log stream.

        Every PutLogEvents request must include the `sequenceToken`
        obtained from the response of the previous request. An upload
        in a newly created log stream does not require a
        `sequenceToken`.

        The batch of events must satisfy the following constraints:

        + The maximum batch size is 32,768 bytes, and this size is
          calculated as the sum of all event messages in UTF-8, plus 26
          bytes for each log event.
        + None of the log events in the batch can be more than 2 hours
          in the future.
        + None of the log events in the batch can be older than 14
          days or the retention period of the log group.
        + The log events in the batch must be in chronological ordered
          by their `timestamp`.
        + The maximum number of log events in a batch is 1,000.

        :type log_group_name: string
        :param log_group_name:

        :type log_stream_name: string
        :param log_stream_name:

        :type log_events: list
        :param log_events: A list of events belonging to a log stream.

        :type sequence_token: string
        :param sequence_token: A string token that must be obtained from the
            response of the previous `PutLogEvents` request.

        """
    params = {'logGroupName': log_group_name, 'logStreamName': log_stream_name, 'logEvents': log_events}
    if sequence_token is not None:
        params['sequenceToken'] = sequence_token
    return self.make_request(action='PutLogEvents', body=json.dumps(params))