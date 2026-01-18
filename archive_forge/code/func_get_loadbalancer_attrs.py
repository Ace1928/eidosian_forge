import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_loadbalancer_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'description': ('description', str), 'protocol': ('protocol', str), 'loadbalancer': ('loadbalancer_id', 'loadbalancers', client_manager.load_balancer.load_balancer_list), 'connection_limit': ('connection_limit', str), 'protocol_port': ('protocol_port', int), 'project': ('project_id', 'project', client_manager.identity), 'vip_address': ('vip_address', str), 'vip_port_id': ('vip_port_id', 'ports', client_manager.neutronclient.list_ports), 'vip_subnet_id': ('vip_subnet_id', 'subnets', client_manager.neutronclient.list_subnets), 'vip_network_id': ('vip_network_id', 'networks', client_manager.neutronclient.list_networks), 'vip_qos_policy_id': ('vip_qos_policy_id', 'policies', client_manager.neutronclient.list_qos_policies), 'vip_vnic_type': ('vip_vnic_type', str), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False), 'cascade': ('cascade', lambda x: True), 'provisioning_status': ('provisioning_status', str), 'operating_status': ('operating_status', str), 'provider': ('provider', str), 'flavor': ('flavor_id', 'flavors', client_manager.load_balancer.flavor_list), 'availability_zone': ('availability_zone', str), 'additional_vip': ('additional_vips', functools.partial(handle_additional_vips, client_manager=client_manager))}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs