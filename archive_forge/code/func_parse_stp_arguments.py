from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import load_config, get_config
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import conditional, remove_default_spec
def parse_stp_arguments(module, item):
    rc, out, err = exec_command(module, 'show interfaces ' + item)
    match = re.search('STP configured to (\\S+),', out, re.S)
    if match:
        return True if match.group(1) == 'ON' else False