import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_create_firewall(self, name, allowed=None, denied=None, network='default', target_ranges=None, direction='INGRESS', priority=1000, source_service_accounts=None, target_service_accounts=None, source_ranges=None, source_tags=None, target_tags=None, description=None):
    """
        Create a firewall rule on a network.
        Rules can be for Ingress or Egress, and they may Allow or
        Deny traffic. They are also applied in order based on action
        (Deny, Allow) and Priority. Rules can be applied using various Source
        and Target filters.

        Firewall rules should be supplied in the "allowed" or "denied" field.
        This is a list of dictionaries formatted like so ("ports" is optional):

            [{"IPProtocol": "<protocol string or number>",
              "ports": "<port_numbers or ranges>"}]

        For example, to allow tcp on port 8080 and udp on all ports, 'allowed'
        would be::

            [{"IPProtocol": "tcp",
              "ports": ["8080"]},
             {"IPProtocol": "udp"}]

        Note that valid inputs vary by direction (INGRESS vs EGRESS), action
        (allow/deny), and source/target filters (tag vs range etc).

        See `Firewall Reference <https://developers.google.com/compute/docs/
        reference/latest/firewalls/insert>`_ for more information.

        :param  name: Name of the firewall to be created
        :type   name: ``str``

        :param  description: Optional description of the rule.
        :type   description: ``str``

        :param  direction: Direction of the FW rule - "INGRESS" or "EGRESS"
                           Defaults to 'INGRESS'.
        :type   direction: ``str``

        :param  priority: Priority integer of the rule -
                          lower is applied first. Defaults to 1000
        :type   priority: ``int``

        :param  allowed: List of dictionaries with rules for type INGRESS
        :type   allowed: ``list`` of ``dict``

        :param  denied: List of dictionaries with rules for type EGRESS
        :type   denied: ``list`` of ``dict``

        :keyword  network: The network that the firewall applies to.
        :type     network: ``str`` or :class:`GCENetwork`

        :keyword  source_ranges: A list of IP ranges in CIDR format that the
                                 firewall should apply to. Defaults to
                                 ['0.0.0.0/0']
        :type     source_ranges: ``list`` of ``str``

        :keyword  source_service_accounts: A list of source service accounts
                                        the rules apply to.
        :type     source_service_accounts: ``list`` of ``str``

        :keyword  source_tags: A list of source instance tags the rules apply
                               to.
        :type     source_tags: ``list`` of ``str``

        :keyword  target_tags: A list of target instance tags the rules apply
                               to.
        :type     target_tags: ``list`` of ``str``

        :keyword  target_service_accounts: A list of target service accounts
                                        the rules apply to.
        :type     target_service_accounts: ``list`` of ``str``

        :keyword  target_ranges: A list of IP ranges in CIDR format that the
                                EGRESS type rule should apply to. Defaults
                                to ['0.0.0.0/0']
        :type     target_ranges: ``list`` of ``str``

        :return:  Firewall object
        :rtype:   :class:`GCEFirewall`
        """
    firewall_data = {}
    if not hasattr(network, 'name'):
        nw = self.ex_get_network(network)
    else:
        nw = network
    firewall_data['name'] = name
    firewall_data['direction'] = direction
    firewall_data['priority'] = priority
    firewall_data['description'] = description
    if direction == 'INGRESS':
        firewall_data['allowed'] = allowed
    elif direction == 'EGRESS':
        firewall_data['denied'] = denied
    firewall_data['network'] = nw.extra['selfLink']
    if source_ranges is None and source_tags is None and (source_service_accounts is None):
        source_ranges = ['0.0.0.0/0']
    if source_ranges is not None:
        firewall_data['sourceRanges'] = source_ranges
    if source_tags is not None:
        firewall_data['sourceTags'] = source_tags
    if source_service_accounts is not None:
        firewall_data['sourceServiceAccounts'] = source_service_accounts
    if target_tags is not None:
        firewall_data['targetTags'] = target_tags
    if target_service_accounts is not None:
        firewall_data['targetServiceAccounts'] = target_service_accounts
    if target_ranges is not None:
        firewall_data['destinationRanges'] = target_ranges
    request = '/global/firewalls'
    self.connection.async_request(request, method='POST', data=firewall_data)
    return self.ex_get_firewall(name)