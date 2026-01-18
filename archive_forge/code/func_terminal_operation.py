from __future__ import absolute_import, division, print_function
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def terminal_operation(module, config):
    if config:
        cmd = 'terminal dont-ask'
    else:
        cmd = 'no terminal dont-ask'
    config_cmd_operation(module, cmd)
    return cmd