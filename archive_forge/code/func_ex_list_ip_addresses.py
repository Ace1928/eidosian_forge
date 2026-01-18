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
def ex_list_ip_addresses(self, node_id):
    """
        List all server addresses.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: :class:`OpenStack_1_0_NodeIpAddresses`
        """
    if isinstance(node_id, Node):
        node_id = node_id.id
    uri = '/servers/%s/ips' % node_id
    resp = self.connection.request(uri, method='GET')
    return self._to_ip_addresses(resp.object)