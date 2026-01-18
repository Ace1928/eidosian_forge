import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_allocate_public_ip(self, vpc_id=None, network_id=None, location=None):
    """
        Allocate a public IP.

        :param vpc_id: VPC the ip belongs to
        :type vpc_id: ``str``

        :param network_id: Network where this IP is connected to.
        :type network_id: ''str''

        :param location: Zone
        :type  location: :class:`NodeLocation`

        :rtype: :class:`CloudStackAddress`
        """
    args = {}
    if location is not None:
        args['zoneid'] = location.id
    else:
        args['zoneid'] = self.list_locations()[0].id
    if vpc_id is not None:
        args['vpcid'] = vpc_id
    if network_id is not None:
        args['networkid'] = network_id
    addr = self._async_request(command='associateIpAddress', params=args, method='GET')
    addr = addr['ipaddress']
    addr = CloudStackAddress(addr['id'], addr['ipaddress'], self)
    return addr