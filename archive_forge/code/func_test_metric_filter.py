import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def test_metric_filter(self, filter_pattern, log_event_messages):
    """
        Tests the filter pattern of a metric filter against a sample
        of log event messages. You can use this operation to validate
        the correctness of a metric filter pattern.

        :type filter_pattern: string
        :param filter_pattern:

        :type log_event_messages: list
        :param log_event_messages:

        """
    params = {'filterPattern': filter_pattern, 'logEventMessages': log_event_messages}
    return self.make_request(action='TestMetricFilter', body=json.dumps(params))