import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def create_log_group(self, log_group_name):
    """
        Creates a new log group with the specified name. The name of
        the log group must be unique within a region for an AWS
        account. You can create up to 100 log groups per account.

        You must use the following guidelines when naming a log group:

        + Log group names can be between 1 and 512 characters long.
        + Allowed characters are az, AZ, 09, '_' (underscore), '-'
          (hyphen), '/' (forward slash), and '.' (period).



        Log groups are created with a default retention of 14 days.
        The retention attribute allow you to configure the number of
        days you want to retain log events in the specified log group.
        See the `SetRetention` operation on how to modify the
        retention of your log groups.

        :type log_group_name: string
        :param log_group_name:

        """
    params = {'logGroupName': log_group_name}
    return self.make_request(action='CreateLogGroup', body=json.dumps(params))