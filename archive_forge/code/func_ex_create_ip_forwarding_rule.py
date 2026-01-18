import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_ip_forwarding_rule(self, node, address, protocol, start_port, end_port=None):
    """
        "Add a NAT/firewall forwarding rule.

        :param      node: Node which should be used
        :type       node: :class:`CloudStackNode`

        :param      address: CloudStackAddress which should be used
        :type       address: :class:`CloudStackAddress`

        :param      protocol: Protocol which should be used (TCP or UDP)
        :type       protocol: ``str``

        :param      start_port: Start port which should be used
        :type       start_port: ``int``

        :param      end_port: End port which should be used
        :type       end_port: ``int``

        :rtype:     :class:`CloudStackForwardingRule`
        """
    protocol = protocol.upper()
    if protocol not in ('TCP', 'UDP'):
        return None
    args = {'ipaddressid': address.id, 'protocol': protocol, 'startport': int(start_port)}
    if end_port is not None:
        args['endport'] = int(end_port)
    result = self._async_request(command='createIpForwardingRule', params=args, method='GET')
    result = result['ipforwardingrule']
    rule = CloudStackIPForwardingRule(node, result['id'], address, protocol, start_port, end_port)
    node.extra['ip_forwarding_rules'].append(rule)
    return rule