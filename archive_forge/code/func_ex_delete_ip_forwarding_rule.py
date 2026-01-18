import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_ip_forwarding_rule(self, node, rule):
    """
        Remove a NAT/firewall forwarding rule.

        :param node: Node which should be used
        :type  node: :class:`CloudStackNode`

        :param rule: Forwarding rule which should be used
        :type  rule: :class:`CloudStackForwardingRule`

        :rtype: ``bool``
        """
    node.extra['ip_forwarding_rules'].remove(rule)
    self._async_request(command='deleteIpForwardingRule', params={'id': rule.id}, method='GET')
    return True