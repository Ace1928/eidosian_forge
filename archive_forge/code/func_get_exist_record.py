from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_connection, rm_config_prefix
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_exist_record(self):
    """get exist netstream record"""
    flags = list()
    exp = ' | ignore-case include netstream record'
    flags.append(exp)
    config = get_config(self.module, flags)
    if config:
        config = config.lstrip()
        config_list = config.split('\n')
        for config_mem in config_list:
            config_mem_list = config_mem.split(' ')
            if len(config_mem_list) > 3 and config_mem_list[3] == 'ip':
                self.existing['ip_record'].append(config_mem_list[2])
            if len(config_mem_list) > 3 and config_mem_list[3] == 'vxlan':
                self.existing['vxlan_record'].append(config_mem_list[2])