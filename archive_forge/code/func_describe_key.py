import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def describe_key(self, key_id):
    """
        Provides detailed information about the specified customer
        master key.

        :type key_id: string
        :param key_id: Unique identifier of the customer master key to be
            described. This can be an ARN, an alias, or a globally unique
            identifier.

        """
    params = {'KeyId': key_id}
    return self.make_request(action='DescribeKey', body=json.dumps(params))