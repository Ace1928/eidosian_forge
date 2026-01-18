import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_ip_address_list(self, ex_network_domain, ex_ip_address_list_name):
    """
        Get IP Address List by name in network domain specified

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
        >>> # Get IP Address List by Name
        >>> ipaddresslist_list_by_name = driver.ex_get_ip_address_list(
        >>>     ex_network_domain=my_network_domain,
        >>>     ex_ip_address_list_name='My_IP_AddressList_1')
        >>> pprint(ipaddresslist_list_by_name)


        :param  ex_network_domain: (required) The network domain or network
                                   domain ID in which ipaddresslist resides.
        :type   ex_network_domain: :class:`NttCisNetworkDomain` or 'str'

        :param    ex_ip_address_list_name: (required) Get 'IP Address List' by
                                            name
        :type     ex_ip_address_list_name: :``str``

        :return: a list of NttCisIpAddressList objects
        :rtype: ``list`` of :class:`NttCisIpAddressList`
        """
    ip_address_lists = self.ex_list_ip_address_list(ex_network_domain)
    return list(filter(lambda x: x.name == ex_ip_address_list_name, ip_address_lists))