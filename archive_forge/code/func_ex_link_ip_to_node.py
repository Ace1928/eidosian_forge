import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_link_ip_to_node(self, node, ip):
    """
        links a existing ip with a node

        :param node: node object
        :type node: ``object``

        :param ip: ip object
        :type ip: ``object``

        :return: Request ID
        :rtype: ``str``
        """
    result = self._sync_request(data={'object_uuid': ip.id}, endpoint='objects/servers/{}/ips/'.format(node.id), method='POST')
    return result