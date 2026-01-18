from __future__ import (absolute_import, division, print_function)
import re
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
def is_interface_support_setjumboframe(interface):
    """is interface support set jumboframe"""
    if interface is None:
        return False
    support_flag = False
    if interface.upper().startswith('GE'):
        support_flag = True
    elif interface.upper().startswith('10GE'):
        support_flag = True
    elif interface.upper().startswith('25GE'):
        support_flag = True
    elif interface.upper().startswith('4X10GE'):
        support_flag = True
    elif interface.upper().startswith('40GE'):
        support_flag = True
    elif interface.upper().startswith('100GE'):
        support_flag = True
    else:
        support_flag = False
    return support_flag