import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_vpn_gateway(self, vpc, for_display=None):
    """
        Creates a VPN Gateway.

        :param vpc: VPC to create the Gateway for (required).
        :type  vpc: :class: `CloudStackVPC`

        :param for_display: Display the VPC to the end user or not.
        :type  for_display: ``bool``

        :rtype: :class: `CloudStackVpnGateway`
        """
    args = {'vpcid': vpc.id}
    if for_display is not None:
        args['fordisplay'] = for_display
    res = self._async_request(command='createVpnGateway', params=args, method='GET')
    item = res['vpngateway']
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['vpngateway']
    return CloudStackVpnGateway(id=item['id'], account=item['account'], domain=item['domain'], domain_id=item['domainid'], public_ip=item['publicip'], vpc_id=vpc.id, driver=self, extra=self._get_extra_dict(item, extra_map))