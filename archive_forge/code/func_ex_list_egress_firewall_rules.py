import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_egress_firewall_rules(self):
    """
        Lists all egress Firewall Rules

        :rtype: ``list`` of :class:`CloudStackEgressFirewallRule`
        """
    rules = []
    result = self._sync_request(command='listEgressFirewallRules', method='GET')
    for rule in result['firewallrule']:
        rules.append(CloudStackEgressFirewallRule(rule['id'], rule['networkid'], rule['cidrlist'], rule['protocol'], rule.get('icmpcode'), rule.get('icmptype'), rule.get('startport'), rule.get('endport')))
    return rules