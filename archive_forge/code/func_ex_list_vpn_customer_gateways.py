import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_vpn_customer_gateways(self, account=None, domain_id=None, id=None, is_recursive=None, keyword=None, list_all=None, page=None, page_size=None, project_id=None):
    """
        List VPN Customer Gateways.

        :param   account: List resources by account (must be
                          used with the domain_id parameter).
        :type    account: ``str``

        :param   domain_id: List only resources belonging
                            to the domain specified.
        :type    domain_id: ``str``

        :param   id: ID of the VPN Customer Gateway.
        :type    id: ``str``

        :param   is_recursive: Defaults to False, but if true, lists all
                               resources from the parent specified by the
                               domain_id till leaves.
        :type    is_recursive: ``bool``

        :param   keyword: List by keyword.
        :type    keyword: ``str``

        :param   list_all: If set to False, list only resources belonging to
                           the command's caller; if set to True - list
                           resources that the caller is authorized to see.
                           Default value is False.
        :type    list_all: ``str``

        :param   page: Start from page.
        :type    page: ``int``

        :param   page_size: Items per page.
        :type    page_size: ``int``

        :param   project_id: List objects by project.
        :type    project_id: ``str``

        :rtype: ``list`` of :class:`CloudStackVpnCustomerGateway`
        """
    args = {}
    if account is not None:
        args['account'] = account
    if domain_id is not None:
        args['domainid'] = domain_id
    if id is not None:
        args['id'] = id
    if is_recursive is not None:
        args['isrecursive'] = is_recursive
    if keyword is not None:
        args['keyword'] = keyword
    if list_all is not None:
        args['listall'] = list_all
    if page is not None:
        args['page'] = page
    if page_size is not None:
        args['pagesize'] = page_size
    if project_id is not None:
        args['projectid'] = project_id
    res = self._sync_request(command='listVpnCustomerGateways', params=args, method='GET')
    items = res.get('vpncustomergateway', [])
    vpn_customer_gateways = []
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['vpncustomergateway']
    for item in items:
        extra = self._get_extra_dict(item, extra_map)
        vpn_customer_gateways.append(CloudStackVpnCustomerGateway(id=item['id'], cidr_list=item['cidrlist'], esp_policy=item['esppolicy'], gateway=item['gateway'], ike_policy=item['ikepolicy'], ipsec_psk=item['ipsecpsk'], driver=self, extra=extra))
    return vpn_customer_gateways