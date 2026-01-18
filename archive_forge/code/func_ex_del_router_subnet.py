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
def ex_del_router_subnet(self, router, subnet):
    """
        Remove subnet to a router

        :param router: Router to remove the subnet
        :type router: :class:`OpenStack_2_Router`

        :param subnet: Subnet object to be added to the router
        :type subnet: :class:`OpenStack_2_SubNet`

        :rtype: ``bool``
        """
    return self._manage_router_interface(router, 'remove', subnet=subnet)