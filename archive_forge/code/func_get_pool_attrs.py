import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_pool_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'description': ('description', str), 'protocol': ('protocol', str), 'pool': ('pool_id', 'pools', client_manager.load_balancer.pool_list), 'loadbalancer': ('loadbalancer_id', 'loadbalancers', client_manager.load_balancer.load_balancer_list), 'lb_algorithm': ('lb_algorithm', str), 'listener': ('listener_id', 'listeners', client_manager.load_balancer.listener_list), 'project': ('project_id', 'project', client_manager.identity), 'session_persistence': ('session_persistence', _format_kv), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False), 'tls_container_ref': ('tls_container_ref', _format_str_if_need_treat_unset), 'ca_tls_container_ref': ('ca_tls_container_ref', _format_str_if_need_treat_unset), 'crl_container_ref': ('crl_container_ref', _format_str_if_need_treat_unset), 'enable_tls': ('tls_enabled', lambda x: True), 'disable_tls': ('tls_enabled', lambda x: False), 'tls_ciphers': ('tls_ciphers', str), 'tls_versions': ('tls_versions', list), 'alpn_protocols': ('alpn_protocols', list)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs