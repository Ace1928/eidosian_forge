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
def ex_set_node_configuration(self, node, **kwargs):
    """
        Update a node configuration.
        Changing most of the parameters requires node to be stopped.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :param      kwargs: keyword arguments
        :type       kwargs: ``dict``

        :rtype: ``bool``
        """
    valid_keys = ('^name$', '^parent$', '^cpu$', '^smp$', '^mem$', '^boot$', '^nic:0:model$', '^nic:0:dhcp', '^nic:1:model$', '^nic:1:vlan$', '^nic:1:mac$', '^vnc:ip$', '^vnc:password$', '^vnc:tls', '^ide:[0-1]:[0-1](:media)?$', '^scsi:0:[0-7](:media)?$', '^block:[0-7](:media)?$')
    invalid_keys = []
    keys = list(kwargs.keys())
    for key in keys:
        matches = False
        for regex in valid_keys:
            if re.match(regex, key):
                matches = True
                break
        if not matches:
            invalid_keys.append(key)
    if invalid_keys:
        raise CloudSigmaException('Invalid configuration key specified: %s' % ','.join(invalid_keys))
    response = self.connection.request(action='/servers/%s/set' % node.id, data=dict2str(kwargs), method='POST')
    return response.status == 200 and response.body != ''