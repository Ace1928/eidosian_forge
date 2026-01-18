from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.snmp_server.snmp_server import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.snmp_server import (
def host_traps_string_to_list(self, hosts):
    if hosts:
        for element in hosts:
            if element.get('traps', {}):
                element['traps'] = list(element.get('traps').split())
        return hosts