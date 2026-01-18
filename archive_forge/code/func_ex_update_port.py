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
def ex_update_port(self, port, description=None, admin_state_up=None, name=None, port_security_enabled=None, qos_policy_id=None, security_groups=None, allowed_address_pairs=None):
    """
        Update a OpenStack_2_PortInterface

        :param      port: port interface to update
        :type       port: :class:`OpenStack_2_PortInterface`

        :param      description: Description of the port
        :type       description: ``str``

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: ``bool``

        :param      name: Human-readable name of the resource
        :type       name: ``str``

        :param      port_security_enabled: 	The port security status
        :type       port_security_enabled: ``bool``

        :param      qos_policy_id: QoS policy associated with the port
        :type       qos_policy_id: ``str``

        :param      security_groups: The IDs of security groups applied
        :type       security_groups: ``list`` of ``str``

        :param      allowed_address_pairs: IP and MAC address that the port
                    can use when sending packets if port_security_enabled is
                    true
        :type       allowed_address_pairs: ``list`` of ``dict`` containing
                    ip_address and mac_address; mac_address is optional, taken
                    from the port if not specified

        :rtype: :class:`OpenStack_2_PortInterface`
        """
    data = {'port': {}}
    if description is not None:
        data['port']['description'] = description
    if admin_state_up is not None:
        data['port']['admin_state_up'] = admin_state_up
    if name is not None:
        data['port']['name'] = name
    if port_security_enabled is not None:
        data['port']['port_security_enabled'] = port_security_enabled
    if qos_policy_id is not None:
        data['port']['qos_policy_id'] = qos_policy_id
    if security_groups is not None:
        data['port']['security_groups'] = security_groups
    if allowed_address_pairs is not None:
        data['port']['allowed_address_pairs'] = allowed_address_pairs
    response = self.network_connection.request('/v2.0/ports/{}'.format(port.id), method='PUT', data=data)
    return self._to_port(response.object['port'])