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
def ex_set_password(self, node, password):
    """
        Changes the administrator password for a specified server.

        :param      node: Node to rebuild.
        :type       node: :class:`Node`

        :param      password: The administrator password.
        :type       password: ``str``

        :rtype: ``bool``
        """
    resp = self._node_action(node, 'changePassword', adminPass=password)
    node.extra['password'] = password
    return resp.status == httplib.ACCEPTED