import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_egress_firewall_rule(self, network_id, cidr_list, protocol, icmp_code=None, icmp_type=None, start_port=None, end_port=None):
    """
        Creates a Firewall Rule

        :param      network_id: the id network network for the egress firewall
                    services
        :type       network_id: ``str``

        :param      cidr_list: cidr list
        :type       cidr_list: ``str``

        :param      protocol: TCP/IP Protocol (TCP, UDP)
        :type       protocol: ``str``

        :param      icmp_code: Error code for this icmp message
        :type       icmp_code: ``int``

        :param      icmp_type: Type of the icmp message being sent
        :type       icmp_type: ``int``

        :param      start_port: start of port range
        :type       start_port: ``int``

        :param      end_port: end of port range
        :type       end_port: ``int``

        :rtype: :class:`CloudStackEgressFirewallRule`
        """
    args = {'networkid': network_id, 'cidrlist': cidr_list, 'protocol': protocol}
    if icmp_code is not None:
        args['icmpcode'] = int(icmp_code)
    if icmp_type is not None:
        args['icmptype'] = int(icmp_type)
    if start_port is not None:
        args['startport'] = int(start_port)
    if end_port is not None:
        args['endport'] = int(end_port)
    result = self._async_request(command='createEgressFirewallRule', params=args, method='GET')
    rule = CloudStackEgressFirewallRule(result['firewallrule']['id'], network_id, cidr_list, protocol, icmp_code, icmp_type, start_port, end_port)
    return rule