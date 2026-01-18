from __future__ import (absolute_import, division, print_function)
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def check_config_if_same(self):
    """Judge whether current config is the same as what we excepted"""
    if self.enable and self.enable != self.dldp_conf['dldpEnable']:
        return False
    if self.internal and self.internal != self.dldp_conf['dldpInterval']:
        return False
    work_mode = 'normal'
    if self.dldp_conf['dldpWorkMode'] == 'dldpEnhance':
        work_mode = 'enhance'
    if self.work_mode and self.work_mode != work_mode:
        return False
    if self.auth_mode:
        if self.auth_mode != 'none':
            return False
        if self.auth_mode == 'none' and self.dldp_conf['dldpAuthMode'] != 'dldpAuthNone':
            return False
    if self.reset and self.reset == 'enable':
        return False
    return True