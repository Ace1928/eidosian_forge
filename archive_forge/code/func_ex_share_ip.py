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
def ex_share_ip(self, group_id, node_id, ip, configure_node=True):
    """
        Shares an IP address to the specified server.

        :param       group_id:  group id which should be used
        :type        group_id: ``str``

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :param       ip: ip which should be used
        :type        ip: ``str``

        :param       configure_node: configure node
        :type        configure_node: ``bool``

        :rtype: ``bool``
        """
    if isinstance(node_id, Node):
        node_id = node_id.id
    if configure_node:
        str_configure = 'true'
    else:
        str_configure = 'false'
    elm = ET.Element('shareIp', {'xmlns': self.XML_NAMESPACE, 'sharedIpGroupId': group_id, 'configureServer': str_configure})
    uri = '/servers/{}/ips/public/{}'.format(node_id, ip)
    resp = self.connection.request(uri, method='PUT', data=ET.tostring(elm))
    return resp.status == httplib.ACCEPTED