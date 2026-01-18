from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.slxos.slxos import get_config, load_config, run_commands
def interface_is_portchannel(name, module):
    if get_interface_type(name) == 'ethernet':
        config = get_config(module)
        if 'channel group' in config:
            return True
    return False