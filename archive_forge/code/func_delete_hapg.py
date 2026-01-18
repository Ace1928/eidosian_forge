import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def delete_hapg(self, hapg_arn):
    """
        Deletes a high-availability partition group.

        :type hapg_arn: string
        :param hapg_arn: The ARN of the high-availability partition group to
            delete.

        """
    params = {'HapgArn': hapg_arn}
    return self.make_request(action='DeleteHapg', body=json.dumps(params))