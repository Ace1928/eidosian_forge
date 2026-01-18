import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_rescue(self, node, password=None):
    """
        Rescue a node

        :param      node: node
        :type       node: :class:`Node`

        :param      password: password
        :type       password: ``str``

        :rtype: :class:`Node`
        """
    if password:
        resp = self._node_action(node, 'rescue', adminPass=password)
    else:
        resp = self._node_action(node, 'rescue')
        password = json.loads(resp.body)['adminPass']
    node.extra['password'] = password
    return node