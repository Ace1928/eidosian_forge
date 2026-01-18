import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_nat_rules(self, network_domain):
    """
        Get NAT rules for the network domain

        :param  network_domain: The network domain the rules belongs to
        :type   network_domain: :class:`NttCisNetworkDomain`

        :rtype: ``list`` of :class:`NttCisNatRule`
        """
    params = {}
    params['networkDomainId'] = network_domain.id
    response = self.connection.request_with_orgId_api_2('network/natRule', params=params).object
    return self._to_nat_rules(response, network_domain)