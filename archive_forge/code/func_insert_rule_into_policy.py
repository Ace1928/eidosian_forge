import typing as ty
from openstack import exceptions
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import (
from openstack.network.v2 import availability_zone
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import extension
from openstack.network.v2 import firewall_group as _firewall_group
from openstack.network.v2 import firewall_policy as _firewall_policy
from openstack.network.v2 import firewall_rule as _firewall_rule
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import floating_ip as _floating_ip
from openstack.network.v2 import health_monitor as _health_monitor
from openstack.network.v2 import l3_conntrack_helper as _l3_conntrack_helper
from openstack.network.v2 import listener as _listener
from openstack.network.v2 import load_balancer as _load_balancer
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import metering_label as _metering_label
from openstack.network.v2 import metering_label_rule as _metering_label_rule
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability
from openstack.network.v2 import (
from openstack.network.v2 import pool as _pool
from openstack.network.v2 import pool_member as _pool_member
from openstack.network.v2 import port as _port
from openstack.network.v2 import port_forwarding as _port_forwarding
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import (
from openstack.network.v2 import qos_policy as _qos_policy
from openstack.network.v2 import qos_rule_type as _qos_rule_type
from openstack.network.v2 import quota as _quota
from openstack.network.v2 import rbac_policy as _rbac_policy
from openstack.network.v2 import router as _router
from openstack.network.v2 import security_group as _security_group
from openstack.network.v2 import security_group_rule as _security_group_rule
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _service_profile
from openstack.network.v2 import service_provider as _service_provider
from openstack.network.v2 import sfc_flow_classifier as _sfc_flow_classifier
from openstack.network.v2 import sfc_port_chain as _sfc_port_chain
from openstack.network.v2 import sfc_port_pair as _sfc_port_pair
from openstack.network.v2 import sfc_port_pair_group as _sfc_port_pair_group
from openstack.network.v2 import sfc_service_graph as _sfc_sservice_graph
from openstack.network.v2 import subnet as _subnet
from openstack.network.v2 import subnet_pool as _subnet_pool
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.network.v2 import trunk as _trunk
from openstack.network.v2 import vpn_endpoint_group as _vpn_endpoint_group
from openstack.network.v2 import vpn_ike_policy as _ike_policy
from openstack.network.v2 import vpn_ipsec_policy as _ipsec_policy
from openstack.network.v2 import (
from openstack.network.v2 import vpn_service as _vpn_service
from openstack import proxy
from openstack import resource
def insert_rule_into_policy(self, firewall_policy_id, firewall_rule_id, insert_after=None, insert_before=None):
    """Insert a firewall_rule into a firewall_policy in order

        :param firewall_policy_id: The ID of the firewall policy.
        :param firewall_rule_id: The ID of the firewall rule.
        :param insert_after: The ID of the firewall rule to insert the new
            rule after. It will be worked only when
            insert_before is none.
        :param insert_before: The ID of the firewall rule to insert the new
            rule before.

        :returns: The updated firewall policy
        :rtype: :class:`~openstack.network.v2.firewall_policy.FirewallPolicy`
        """
    body = {'firewall_rule_id': firewall_rule_id, 'insert_after': insert_after, 'insert_before': insert_before}
    policy = self._get_resource(_firewall_policy.FirewallPolicy, firewall_policy_id)
    return policy.insert_rule(self, **body)