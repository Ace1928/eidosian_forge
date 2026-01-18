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
def ex_create_port(self, network, description=None, admin_state_up=True, name=None):
    """
        Creates a new OpenStack_2_PortInterface

        :param      network: ID of the network where the newly created
                    port should be attached to
        :type       network: :class:`OpenStackNetwork`

        :param      description: Description of the port
        :type       description: str

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: bool

        :param      name: Human-readable name of the resource
        :type       name: str

        :rtype: :class:`OpenStack_2_PortInterface`
        """
    data = {'port': {'description': description or '', 'admin_state_up': admin_state_up, 'name': name or '', 'network_id': network.id}}
    response = self.network_connection.request('/v2.0/ports', method='POST', data=data)
    return self._to_port(response.object['port'])