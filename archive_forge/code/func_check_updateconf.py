from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def check_updateconf(module, to_check):
    """Checks if attributes are compatible with one.vm.updateconf API call."""
    for attr, subattributes in to_check.items():
        if attr not in UPDATECONF_ATTRIBUTES:
            module.fail_json(msg="'{0:}' is not a valid VM attribute.".format(attr))
        if not UPDATECONF_ATTRIBUTES[attr]:
            continue
        for subattr in subattributes:
            if subattr not in UPDATECONF_ATTRIBUTES[attr]:
                module.fail_json(msg="'{0:}' is not a valid VM subattribute of '{1:}'".format(subattr, attr))