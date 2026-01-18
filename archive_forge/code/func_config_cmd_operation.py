from __future__ import absolute_import, division, print_function
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_cmd_operation(module, cmd):
    iteration = 0
    while iteration < 10:
        msg = load_config(module, [cmd], True)
        if msg:
            if 'another install operation is in progress' in msg[0].lower() or 'failed' in msg[0].lower():
                time.sleep(2)
                iteration += 1
            else:
                return
        else:
            return