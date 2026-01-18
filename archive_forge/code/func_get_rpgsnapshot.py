from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def get_rpgsnapshot(module, array):
    """Return Replicated Snapshot or None"""
    try:
        snapname = module.params['name'] + '.' + module.params['suffix'] + '.' + module.params['restore']
        array.get_volume_snapshots(names=[snapname])
        return snapname
    except AttributeError:
        return None