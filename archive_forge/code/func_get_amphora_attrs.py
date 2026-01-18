import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_amphora_attrs(client_manager, parsed_args):
    attr_map = {'amphora_id': ('amphora_id', 'amphorae', client_manager.load_balancer.amphora_list), 'loadbalancer': ('loadbalancer_id', 'loadbalancers', client_manager.load_balancer.load_balancer_list), 'compute_id': ('compute_id', str), 'role': ('role', str), 'status': ('status', str)}
    return _map_attrs(vars(parsed_args), attr_map)