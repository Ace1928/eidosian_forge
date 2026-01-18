import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_delete_floating_ip(self, ip):
    """
        Delete specified floating IP

        :param      ip: floating IP to remove
        :type       ip: :class:`DigitalOcean_v2_FloatingIpAddress`

        :rtype: ``bool``
        """
    resp = self.connection.request('/v2/floating_ips/{}'.format(ip.id), method='DELETE')
    return resp.status == httplib.NO_CONTENT