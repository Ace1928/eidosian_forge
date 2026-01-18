import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_nat_rule(self, network_domain, rule_id):
    """
        Get a NAT rule by ID

        :param  network_domain: The network domain the rule belongs to
        :type   network_domain: :class:`NttCisNetworkDomain`

        :param  rule_id: The ID of the NAT rule to fetch
        :type   rule_id: ``str``

        :rtype: :class:`NttCisNatRule`
        """
    rule = self.connection.request_with_orgId_api_2('network/natRule/%s' % rule_id).object
    return self._to_nat_rule(rule, network_domain)