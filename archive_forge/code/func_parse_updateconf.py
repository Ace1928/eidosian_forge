from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def parse_updateconf(vm_template):
    """Extracts 'updateconf' attributes from a VM template."""
    updateconf = {}
    for attr, subattributes in vm_template.items():
        if attr not in UPDATECONF_ATTRIBUTES:
            continue
        tmp = {}
        for subattr, value in subattributes.items():
            if UPDATECONF_ATTRIBUTES[attr] and subattr not in UPDATECONF_ATTRIBUTES[attr]:
                continue
            tmp[subattr] = value
        if tmp:
            updateconf[attr] = tmp
    return updateconf