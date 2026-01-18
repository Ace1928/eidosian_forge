import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def delete_log_group(self, log_group_name):
    """
        Deletes the log group with the specified name. Amazon
        CloudWatch Logs will delete a log group only if there are no
        log streams and no metric filters associated with the log
        group. If this condition is not satisfied, the request will
        fail and the log group will not be deleted.

        :type log_group_name: string
        :param log_group_name:

        """
    params = {'logGroupName': log_group_name}
    return self.make_request(action='DeleteLogGroup', body=json.dumps(params))