from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acl_interfaces.acl_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acl_interfaces import (
def get_acl_interfaces_data(self, connection):
    return connection.get('show running-config | include ^interface|ip access-group|ipv6 traffic-filter')