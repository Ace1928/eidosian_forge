from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def get_destroyed_endpoint(vol, array):
    """Return Destroyed Endpoint or None"""
    try:
        return bool(array.get_volume(vol, protocol_endpoint=True, pending=True)['time_remaining'] != '')
    except Exception:
        return False