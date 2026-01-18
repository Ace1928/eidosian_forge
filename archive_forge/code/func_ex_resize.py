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
def ex_resize(self, node, size):
    """
        Change a node size.

        :param      node: Node to resize.
        :type       node: :class:`Node`

        :type       size: :class:`NodeSize`
        :param      size: New size to use.

        :rtype: ``bool``
        """
    server_params = {'flavorRef': size.id}
    resp = self._node_action(node, 'resize', **server_params)
    return resp.status == httplib.ACCEPTED