from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def get_local_pod(module, array):
    """Return Pod or None"""
    try:
        return array.get_pod(module.params['name'])
    except Exception:
        return None