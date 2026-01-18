import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_l7rule_attrs(client_manager, parsed_args):
    attr_map = {'action': ('action', str), 'project': ('project_id', 'project', client_manager.identity), 'invert': ('invert', lambda x: True), 'l7rule': ('l7rule_id', 'l7rules', 'l7policy', client_manager.load_balancer.l7rule_list), 'l7policy': ('l7policy_id', 'l7policies', client_manager.load_balancer.l7policy_list), 'value': ('value', str), 'key': ('key', str), 'type': ('type', str), 'compare_type': ('compare_type', str), 'enable': ('admin_state_up', lambda x: True), 'disable': ('admin_state_up', lambda x: False)}
    add_tags_attr_map(attr_map)
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs