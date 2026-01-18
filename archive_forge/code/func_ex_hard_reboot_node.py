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
def ex_hard_reboot_node(self, node):
    """
        Hard reboots the specified server

        :param      node:  node
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
    return self._reboot_node(node, reboot_type='HARD')