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
def ex_remove_security_group_from_node(self, security_group, node):
    """
        Remove a Security Group from a node.

        :param security_group: Security Group to remove from node.
        :type  security_group: :class:`OpenStackSecurityGroup`

        :param      node: Node to remove the Security Group.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
    server_params = {'name': security_group.name}
    resp = self._node_action(node, 'removeSecurityGroup', **server_params)
    return resp.status == httplib.ACCEPTED