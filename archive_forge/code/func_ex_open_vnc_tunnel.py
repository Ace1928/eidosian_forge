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
def ex_open_vnc_tunnel(self, node):
    """
        Open a VNC tunnel to the provided node and return the VNC url.

        :param node: Node to open the VNC tunnel to.
        :type node: :class:`libcloud.compute.base.Node`

        :return: URL of the opened VNC tunnel.
        :rtype: ``str``
        """
    path = '/servers/%s/action/' % node.id
    response = self._perform_action(path=path, action='open_vnc', method='POST').object
    vnc_url = response['vnc_url']
    return vnc_url