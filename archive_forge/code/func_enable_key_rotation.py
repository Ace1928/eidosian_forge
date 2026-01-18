import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def enable_key_rotation(self, key_id):
    """
        Enables rotation of the specified customer master key.

        :type key_id: string
        :param key_id: Unique identifier of the customer master key for which
            rotation is to be enabled. This can be an ARN, an alias, or a
            globally unique identifier.

        """
    params = {'KeyId': key_id}
    return self.make_request(action='EnableKeyRotation', body=json.dumps(params))