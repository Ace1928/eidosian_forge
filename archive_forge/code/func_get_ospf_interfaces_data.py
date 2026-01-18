from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ospf_interfaces.ospf_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospf_interfaces import (
def get_ospf_interfaces_data(self, connection):
    return connection.get('show running-config | section ^interface')