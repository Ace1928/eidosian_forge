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
def ex_list_floating_ip_pools(self):
    """
        List available floating IP pools

        :rtype: ``list`` of :class:`OpenStack_2_FloatingIpPool`
        """
    return self._to_floating_ip_pools(self.network_connection.request('/v2.0/networks?router:external=True&fields=id&fields=name').object)