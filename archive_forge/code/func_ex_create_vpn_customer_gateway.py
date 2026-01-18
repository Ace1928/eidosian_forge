import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_vpn_customer_gateway(self, cidr_list, esp_policy, gateway, ike_policy, ipsec_psk, account=None, domain_id=None, dpd=None, esp_lifetime=None, ike_lifetime=None, name=None):
    """
        Creates a VPN Customer Gateway.

        :param cidr_list: Guest CIDR list of the Customer Gateway (required).
        :type  cidr_list: ``str``

        :param esp_policy: ESP policy of the Customer Gateway (required).
        :type  esp_policy: ``str``

        :param gateway: Public IP address of the Customer Gateway (required).
        :type  gateway: ``str``

        :param ike_policy: IKE policy of the Customer Gateway (required).
        :type  ike_policy: ``str``

        :param ipsec_psk: IPsec preshared-key of the Customer Gateway
                          (required).
        :type  ipsec_psk: ``str``

        :param account: The associated account with the Customer Gateway
                        (must be used with the domain_id param).
        :type  account: ``str``

        :param domain_id: The domain ID associated with the Customer Gateway.
                          If used with the account parameter returns the
                          gateway associated with the account for the
                          specified domain.
        :type  domain_id: ``str``

        :param dpd: If DPD is enabled for the VPN connection.
        :type  dpd: ``bool``

        :param esp_lifetime: Lifetime of phase 2 VPN connection to the
                             Customer Gateway, in seconds.
        :type  esp_lifetime: ``int``

        :param ike_lifetime: Lifetime of phase 1 VPN connection to the
                             Customer Gateway, in seconds.
        :type  ike_lifetime: ``int``

        :param name: Name of the Customer Gateway.
        :type  name: ``str``

        :rtype: :class: `CloudStackVpnCustomerGateway`
        """
    args = {'cidrlist': cidr_list, 'esppolicy': esp_policy, 'gateway': gateway, 'ikepolicy': ike_policy, 'ipsecpsk': ipsec_psk}
    if account is not None:
        args['account'] = account
    if domain_id is not None:
        args['domainid'] = domain_id
    if dpd is not None:
        args['dpd'] = dpd
    if esp_lifetime is not None:
        args['esplifetime'] = esp_lifetime
    if ike_lifetime is not None:
        args['ikelifetime'] = ike_lifetime
    if name is not None:
        args['name'] = name
    res = self._async_request(command='createVpnCustomerGateway', params=args, method='GET')
    item = res['vpncustomergateway']
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['vpncustomergateway']
    return CloudStackVpnCustomerGateway(id=item['id'], cidr_list=cidr_list, esp_policy=esp_policy, gateway=gateway, ike_policy=ike_policy, ipsec_psk=ipsec_psk, driver=self, extra=self._get_extra_dict(item, extra_map))