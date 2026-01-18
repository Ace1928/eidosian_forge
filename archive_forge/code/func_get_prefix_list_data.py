from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.prefix_lists.prefix_lists import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.prefix_lists import (
def get_prefix_list_data(self, connection):
    return connection.get('show running-config | section ^ip prefix-list|^ipv6 prefix-list')