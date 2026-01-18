from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def is_config_exist(cmp_cfg, test_cfg):
    """is configuration exist"""
    test_cfg_tmp = test_cfg + ' *$' + '|' + test_cfg + ' *\n'
    obj = re.compile(test_cfg_tmp)
    result = re.findall(obj, cmp_cfg)
    if not result:
        return False
    return True