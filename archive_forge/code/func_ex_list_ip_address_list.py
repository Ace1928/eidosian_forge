import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_ip_address_list(self, ex_network_domain):
    """
        List IP Address List by network domain ID specified

        >>> from pprint import pprint
        >>> from libcloud.compute.types import Provider
        >>> from libcloud.compute.providers import get_driver
        >>> import libcloud.security
        >>>
        >>> # Get NTTC-CIS driver
        >>> libcloud.security.VERIFY_SSL_CERT = True
        >>> cls = get_driver(Provider.NTTCIS)
        >>> driver = cls('myusername','mypassword', region='dd-au')
        >>>
        >>> # Get location
        >>> location = driver.ex_get_location_by_id(id='AU9')
        >>>
        >>> # Get network domain by location
        >>> networkDomainName = "Baas QA"
        >>> network_domains = driver.ex_list_network_domains(location=location)
        >>> my_network_domain = [d for d in network_domains if d.name ==
                              networkDomainName][0]
        >>>
        >>> # List IP Address List of network domain
        >>> ipaddresslist_list = driver.ex_list_ip_address_list(
        >>>     ex_network_domain=my_network_domain)
        >>> pprint(ipaddresslist_list)

        :param  ex_network_domain: The network domain or network domain ID
        :type   ex_network_domain: :class:`NttCisNetworkDomain` or 'str'

        :return: a list of NttCisIpAddressList objects
        :rtype: ``list`` of :class:`NttCisIpAddressList`
        """
    params = {'networkDomainId': self._network_domain_to_network_domain_id(ex_network_domain)}
    response = self.connection.request_with_orgId_api_2('network/ipAddressList', params=params).object
    return self._to_ip_address_lists(response)