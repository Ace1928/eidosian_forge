from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.static_routes.static_routes import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def get_afi(self, address):
    route_type = get_route_type(address)
    if route_type == 'route':
        return 'ipv4'
    elif route_type == 'route6':
        return 'ipv6'