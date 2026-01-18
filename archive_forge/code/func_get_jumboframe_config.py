from __future__ import (absolute_import, division, print_function)
import re
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def get_jumboframe_config(self):
    """ get_jumboframe_config"""
    flags = list()
    exp = '| ignore-case section include ^#\\s+interface %s\\s+' % self.interface.replace(' ', '')
    flags.append(exp)
    output = self.get_config(flags)
    output = output.replace('*', '').lower()
    return self.prase_jumboframe_para(output)