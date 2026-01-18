import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_vpn_customer_gateway(self, vpn_customer_gateway):
    """
        Deletes a VPN Customer Gateway.

        :param vpn_customer_gateway: The VPN Customer Gateway (required).
        :type  vpn_customer_gateway: :class:`CloudStackVpnCustomerGateway`

        :rtype: ``bool``
        """
    res = self._async_request(command='deleteVpnCustomerGateway', params={'id': vpn_customer_gateway.id}, method='GET')
    return res['success']