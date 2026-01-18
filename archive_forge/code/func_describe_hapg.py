import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def describe_hapg(self, hapg_arn):
    """
        Retrieves information about a high-availability partition
        group.

        :type hapg_arn: string
        :param hapg_arn: The ARN of the high-availability partition group to
            describe.

        """
    params = {'HapgArn': hapg_arn}
    return self.make_request(action='DescribeHapg', body=json.dumps(params))