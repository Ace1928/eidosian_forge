import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_flavor_attrs(client_manager, parsed_args):
    attr_map = {'name': ('name', str), 'flavor': ('flavor_id', 'flavors', client_manager.load_balancer.flavor_list), 'flavorprofile': ('flavor_profile_id', 'flavorprofiles', client_manager.load_balancer.flavorprofile_list), 'enable': ('enabled', lambda x: True), 'disable': ('enabled', lambda x: False), 'description': ('description', str)}
    _attrs = vars(parsed_args)
    attrs = _map_attrs(_attrs, attr_map)
    return attrs