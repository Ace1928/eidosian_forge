from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_vgroup(module, array):
    """Get Volume Group"""
    vgroup = None
    for vgrp in array.list_vgroups():
        if vgrp['name'] == module.params['name']:
            vgroup = vgrp
            break
    return vgroup