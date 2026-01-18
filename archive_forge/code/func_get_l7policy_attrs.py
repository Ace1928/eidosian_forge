import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_l7policy_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'description': ('description', str), 'redirect_url': ('redirect_url', str), 'redirect_http_code': ('redirect_http_code', int), 'redirect_prefix': ('redirect_prefix', str), 'l7policy': ('l7policy_id', 'l7policies', client_manager.load_balancer.l7policy_list), 'redirect_pool': ('redirect_pool_id', 'pools', client_manager.load_balancer.pool_list), 'listener': ('listener_id', 'listeners', client_manager.load_balancer.listener_list), 'action': ('action', str), 'project': ('project_id', 'projects', client_manager.identity), 'position': ('position', int), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs