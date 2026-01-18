import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_get_floating_ip(self, ip):
    """
        Get specified floating IP

        :param      ip: floating IP to get
        :type       ip: ``str``

        :rtype: :class:`DigitalOcean_v2_FloatingIpAddress`
        """
    floating_ips = self.ex_list_floating_ips()
    matching_ips = [x for x in floating_ips if x.ip_address == ip]
    if not matching_ips:
        raise ValueError('Floating ip %s not found' % ip)
    return matching_ips[0]