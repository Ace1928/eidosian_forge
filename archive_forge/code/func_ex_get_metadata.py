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
def ex_get_metadata(self, node):
    """
        Get a Node's metadata.

        :param      node: Node
        :type       node: :class:`Node`

        :return: Key/Value metadata associated with node.
        :rtype: ``dict``
        """
    return self.connection.request('/servers/{}/metadata'.format(node.id), method='GET').object['metadata']