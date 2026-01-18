import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_attach_floating_ip_to_node(self, node, ip):
    """
        Attach the floating IP to the node

        :param      node: node
        :type       node: :class:`Node`

        :param      ip: floating IP to attach
        :type       ip: ``str`` or :class:`DigitalOcean_v2_FloatingIpAddress`

        :rtype: ``bool``
        """
    data = {'type': 'assign', 'droplet_id': node.id}
    resp = self.connection.request('/v2/floating_ips/%s/actions' % ip.ip_address, data=json.dumps(data), method='POST')
    return resp.status == httplib.CREATED