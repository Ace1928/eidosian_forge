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
def ex_close_vnc_tunnel(self, node):
    """
        Close a VNC server to the provided node.

        :param node: Node to close the VNC tunnel to.
        :type node: :class:`libcloud.compute.base.Node`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
    path = '/servers/%s/action/' % node.id
    response = self._perform_action(path=path, action='close_vnc', method='POST')
    return response.status == httplib.ACCEPTED