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
def ex_unshare_ip(self, node_id, ip):
    """
        Removes a shared IP address from the specified server.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :param       ip: ip which should be used
        :type        ip: ``str``

        :rtype: ``bool``
        """
    if isinstance(node_id, Node):
        node_id = node_id.id
    uri = '/servers/{}/ips/public/{}'.format(node_id, ip)
    resp = self.connection.request(uri, method='DELETE')
    return resp.status == httplib.ACCEPTED