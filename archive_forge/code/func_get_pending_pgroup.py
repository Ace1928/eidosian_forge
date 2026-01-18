from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_pending_pgroup(module, array):
    """Get Protection Group"""
    pgroup = None
    if ':' in module.params['pgroup']:
        if '::' not in module.params['pgroup']:
            for pgrp in array.list_pgroups(pending=True, on='*'):
                if pgrp['name'] == module.params['pgroup'] and pgrp['time_remaining']:
                    pgroup = pgrp
                    break
        else:
            for pgrp in array.list_pgroups(pending=True):
                if pgrp['name'] == module.params['pgroup'] and pgrp['time_remaining']:
                    pgroup = pgrp
                    break
    else:
        for pgrp in array.list_pgroups(pending=True):
            if pgrp['name'] == module.params['pgroup'] and pgrp['time_remaining']:
                pgroup = pgrp
                break
    return pgroup