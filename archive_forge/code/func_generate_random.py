import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def generate_random(self, number_of_bytes=None):
    """
        Generates an unpredictable byte string.

        :type number_of_bytes: integer
        :param number_of_bytes: Integer that contains the number of bytes to
            generate. Common values are 128, 256, 512, 1024 and so on. The
            current limit is 1024 bytes.

        """
    params = {}
    if number_of_bytes is not None:
        params['NumberOfBytes'] = number_of_bytes
    response = self.make_request(action='GenerateRandom', body=json.dumps(params))
    if response.get('Plaintext') is not None:
        response['Plaintext'] = base64.b64decode(response['Plaintext'].encode('utf-8'))
    return response