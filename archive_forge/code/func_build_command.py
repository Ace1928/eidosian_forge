from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def build_command(param_dict, command):
    for param in ['group_list', 'prefix_list', 'route_map']:
        if param_dict.get(param):
            command += ' {0} {1}'.format(param.replace('_', '-'), param_dict.get(param))
    if param_dict.get('bidir'):
        command += ' bidir'
    return [command]