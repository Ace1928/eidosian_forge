from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_destroyed_target(module, array):
    """Return Destroyed Volume or None"""
    try:
        return bool(array.get_pod(module.params['target'], pending=True)['time_remaining'] != '')
    except Exception:
        return False