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
def ex_delete_port(self, port):
    """
        Delete an OpenStack_2_PortInterface

        https://developer.openstack.org/api-ref/network/v2/#delete-port

        :param      port: port interface to remove
        :type       port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
    response = self.network_connection.request('/v2.0/ports/%s' % port.id, method='DELETE')
    return response.success()