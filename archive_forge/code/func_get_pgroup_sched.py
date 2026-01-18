from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def get_pgroup_sched(module, array):
    """Get Protection Group Schedule"""
    pgroup = None
    for pgrp in array.list_pgroups(schedule=True):
        if pgrp['name'].casefold() == module.params['name'].casefold():
            pgroup = pgrp
            break
    return pgroup