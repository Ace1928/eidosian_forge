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
def ex_attach_port_interface(self, node, port):
    """
        Attaches an OpenStack_2_PortInterface to a Node.

        :param      node: node
        :type       node: :class:`Node`

        :param      port: port interface to attach
        :type       port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
    data = {'interfaceAttachment': {'port_id': port.id}}
    return self.connection.request('/servers/{}/os-interface'.format(node.id), method='POST', data=data).success()