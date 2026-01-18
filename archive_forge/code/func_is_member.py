from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.connection import ConnectionError, exec_command
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import run_commands, get_config, load_config
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import CustomNetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
def is_member(member, lst):
    for li in lst:
        ml = range_to_members(li)
        if member in ml:
            return True
    return False