import argparse
import copy
from random import choice
from random import randint
from unittest import mock
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group as _address_group
from openstack.network.v2 import address_scope as _address_scope
from openstack.network.v2 import agent as network_agent
from openstack.network.v2 import auto_allocated_topology as allocated_topology
from openstack.network.v2 import availability_zone as _availability_zone
from openstack.network.v2 import extension as _extension
from openstack.network.v2 import flavor as _flavor
from openstack.network.v2 import local_ip as _local_ip
from openstack.network.v2 import local_ip_association as _local_ip_association
from openstack.network.v2 import ndp_proxy as _ndp_proxy
from openstack.network.v2 import network as _network
from openstack.network.v2 import network_ip_availability as _ip_availability
from openstack.network.v2 import network_segment_range as _segment_range
from openstack.network.v2 import port as _port
from openstack.network.v2 import rbac_policy as network_rbac
from openstack.network.v2 import segment as _segment
from openstack.network.v2 import service_profile as _flavor_profile
from openstack.network.v2 import trunk as _trunk
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit import utils
def create_one_port(attrs=None):
    """Create a fake port.

    :param Dictionary attrs:
        A dictionary with all attributes
    :return:
        A Port object, with id, name, etc.
    """
    attrs = attrs or {}
    port_attrs = {'is_admin_state_up': True, 'allowed_address_pairs': [{}], 'binding:host_id': 'binding-host-id-' + uuid.uuid4().hex, 'binding:profile': {}, 'binding:vif_details': {}, 'binding:vif_type': 'ovs', 'binding:vnic_type': 'normal', 'data_plane_status': None, 'description': 'description-' + uuid.uuid4().hex, 'device_id': 'device-id-' + uuid.uuid4().hex, 'device_owner': 'compute:nova', 'device_profile': 'cyborg_device_profile_1', 'dns_assignment': [{}], 'dns_domain': 'dns-domain-' + uuid.uuid4().hex, 'dns_name': 'dns-name-' + uuid.uuid4().hex, 'extra_dhcp_opts': [{}], 'fixed_ips': [{'ip_address': '10.0.0.3', 'subnet_id': 'subnet-id-' + uuid.uuid4().hex}], 'hardware_offload_type': None, 'hints': {}, 'id': 'port-id-' + uuid.uuid4().hex, 'mac_address': 'fa:16:3e:a9:4e:72', 'name': 'port-name-' + uuid.uuid4().hex, 'network_id': 'network-id-' + uuid.uuid4().hex, 'numa_affinity_policy': 'required', 'is_port_security_enabled': True, 'security_group_ids': [], 'status': 'ACTIVE', 'project_id': 'project-id-' + uuid.uuid4().hex, 'qos_network_policy_id': 'qos-policy-id-' + uuid.uuid4().hex, 'qos_policy_id': 'qos-policy-id-' + uuid.uuid4().hex, 'tags': [], 'propagate_uplink_status': False, 'location': 'MUNCHMUNCHMUNCH'}
    port_attrs.update(attrs)
    port = _port.Port(**port_attrs)
    return port