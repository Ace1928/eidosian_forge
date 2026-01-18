from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
def get_bd_list(self):
    """get bridge domain list"""
    flags = list()
    bd_info = list()
    exp = ' include-default | include bridge-domain | exclude undo'
    flags.append(exp)
    bd_str = self.get_config(flags)
    if not bd_str:
        return bd_info
    bd_num = re.findall('bridge-domain\\s*([0-9]+)', bd_str)
    bd_info.extend(bd_num)
    return bd_info