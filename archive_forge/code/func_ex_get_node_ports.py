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
def ex_get_node_ports(self, node):
    """
        Get the list of OpenStack_2_PortInterface interfaces from a Node.
        :param      node: node
        :type       node: :class:`Node`

        :rtype: ``list`` of :class:`OpenStack_2_PortInterface`
        """
    response = self.connection.request('/servers/%s/os-interface' % node.id, method='GET')
    ports = []
    for port in response.object['interfaceAttachments']:
        port['id'] = port.pop('port_id')
        ports.append(self._to_port(port))
    return ports