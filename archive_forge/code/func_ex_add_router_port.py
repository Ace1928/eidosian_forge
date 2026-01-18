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
def ex_add_router_port(self, router, port):
    """
        Add port to a router

        :param router: Router to add the port
        :type router: :class:`OpenStack_2_Router`

        :param port: Port object to be added to the router
        :type port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
    return self._manage_router_interface(router, 'add', port=port)