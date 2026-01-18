import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def delete_luna_client(self, client_arn):
    """
        Deletes a client.

        :type client_arn: string
        :param client_arn: The ARN of the client to delete.

        """
    params = {'ClientArn': client_arn}
    return self.make_request(action='DeleteLunaClient', body=json.dumps(params))