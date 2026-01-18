import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def ex_get_console_output(self, node):
    """
        Gets console output for the node.

        :param      node: Node which should be used
        :type       node: :class:`Node`

        :return:    A dictionary with the following keys:
                    - instance_id (``str``)
                    - timestamp (``datetime.datetime``) - last output timestamp
                    - output (``str``) - console output
        :rtype:     ``dict``
        """
    params = {'Action': 'GetConsoleOutput', 'InstanceId': node.id}
    response = self.connection.request(self.path, params=params).object
    timestamp = findattr(element=response, xpath='timestamp', namespace=NAMESPACE)
    encoded_string = findattr(element=response, xpath='output', namespace=NAMESPACE)
    timestamp = parse_date(timestamp)
    if encoded_string:
        output = base64.b64decode(b(encoded_string)).decode('utf-8')
    else:
        output = None
    return {'instance_id': node.id, 'timestamp': timestamp, 'output': output}