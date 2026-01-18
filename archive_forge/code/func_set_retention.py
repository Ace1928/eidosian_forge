import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def set_retention(self, log_group_name, retention_in_days):
    """
        Sets the retention of the specified log group. Log groups are
        created with a default retention of 14 days. The retention
        attribute allow you to configure the number of days you want
        to retain log events in the specified log group.

        :type log_group_name: string
        :param log_group_name:

        :type retention_in_days: integer
        :param retention_in_days: Specifies the number of days you want to
            retain log events in the specified log group. Possible values are:
            1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 547, 730.

        """
    params = {'logGroupName': log_group_name, 'retentionInDays': retention_in_days}
    return self.make_request(action='SetRetention', body=json.dumps(params))