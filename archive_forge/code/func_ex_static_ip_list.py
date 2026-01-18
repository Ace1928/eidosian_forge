import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_static_ip_list(self):
    """
        Return a list of available static IP addresses.

        :rtype: ``list`` of ``str``
        """
    response = self.connection.request(action='/resources/ip/list', method='GET')
    if response.status != 200:
        raise CloudSigmaException('Could not retrieve IP list')
    ips = str2list(response.body)
    return ips