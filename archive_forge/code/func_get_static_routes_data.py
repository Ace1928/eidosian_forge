from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.static_routes.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def get_static_routes_data(self, connection):
    return connection.get('show running-config | include ^ip route .+ |^ipv6 route .+')