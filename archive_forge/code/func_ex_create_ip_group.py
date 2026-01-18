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
def ex_create_ip_group(self, group_name, node_id=None):
    """
        Creates a shared IP group.

        :param       group_name:  group name which should be used
        :type        group_name: ``str``

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: ``bool``
        """
    if isinstance(node_id, Node):
        node_id = node_id.id
    group_elm = ET.Element('sharedIpGroup', {'xmlns': self.XML_NAMESPACE, 'name': group_name})
    if node_id:
        ET.SubElement(group_elm, 'server', {'id': node_id})
    resp = self.connection.request('/shared_ip_groups', method='POST', data=ET.tostring(group_elm))
    return self._to_shared_ip_group(resp.object)